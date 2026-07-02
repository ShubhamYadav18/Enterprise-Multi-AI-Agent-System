"""
Verification test script for the Security Module.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from security.sbom import generate_sbom
from security.utils import is_cdxgen_available, is_npm_available, parse_dependency_tree
from security.defectdojo import DefectDojoClient
from security.service import SecurityService
from security.models import SecurityScanResult


class TestSecurityModule(unittest.TestCase):
    """Test suite for security verification."""

    def test_environment_check(self):
        """Verify that capability check behaves correctly."""
        cdx = is_cdxgen_available()
        npm = is_npm_available()
        print(f"\n[Environment] cdxgen available: {cdx}, npm available: {npm}")
        self.assertTrue(cdx or npm, "At least npm or cdxgen should be available to run tests.")

    def test_sbom_generation(self):
        """Test that CycloneDX SBOM is generated successfully."""
        # Use a temporary output path
        output_file = PROJECT_ROOT / "sbom" / "bom.json"
        
        # Clean up any existing file
        if output_file.exists():
            try:
                os.remove(output_file)
            except Exception:
                pass

        print(f"[SBOM] Running cdxgen to generate: {output_file}...")
        res = generate_sbom(output_path=output_file)
        
        print(f"[SBOM] Result status: {res.status}, duration: {res.duration_ms:.0f}ms")
        self.assertEqual(res.status, "success", f"SBOM generation failed: {res.error}")
        self.assertTrue(output_file.exists(), "SBOM file was not created.")
        self.assertGreater(output_file.stat().st_size, 0, "SBOM file is empty.")

    def test_dependency_tree_parsing(self):
        """Test parsing of generated bom.json into a dependency tree."""
        mock_sbom_data = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "metadata": {
                "component": {
                    "bom-ref": "pkg:pypi/project@1.0.0",
                    "name": "project",
                    "version": "1.0.0"
                }
            },
            "components": [
                {
                    "bom-ref": "pkg:pypi/fastapi@0.115.0",
                    "name": "fastapi",
                    "version": "0.115.0"
                },
                {
                    "bom-ref": "pkg:pypi/starlette@0.27.0",
                    "name": "starlette",
                    "version": "0.27.0"
                }
            ],
            "dependencies": [
                {
                    "ref": "pkg:pypi/project@1.0.0",
                    "dependsOn": ["pkg:pypi/fastapi@0.115.0"]
                },
                {
                    "ref": "pkg:pypi/fastapi@0.115.0",
                    "dependsOn": ["pkg:pypi/starlette@0.27.0"]
                }
            ]
        }
        
        temp_sbom = PROJECT_ROOT / "sbom" / "test_bom.json"
        temp_sbom.parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(temp_sbom, "w", encoding="utf-8") as f:
            json.dump(mock_sbom_data, f)
            
        try:
            print(f"[Tree] Parsing tree from: {temp_sbom}...")
            tree = parse_dependency_tree(temp_sbom)
            
            self.assertIn("name", tree)
            self.assertEqual(tree["name"], "project")
            self.assertIn("dependencies", tree)
            self.assertEqual(len(tree["dependencies"]), 1)
            self.assertEqual(tree["dependencies"][0]["name"], "fastapi")
            self.assertEqual(tree["dependencies"][0]["dependencies"][0]["name"], "starlette")
        finally:
            if temp_sbom.exists():
                os.remove(temp_sbom)


    def test_offline_scan_fallback(self):
        """Test that SecurityService runs in offline mode when credentials are empty."""
        print("[Service] Testing SecurityService offline fallback...")
        service = SecurityService()
        
        # Force settings to have empty DefectDojo values
        with patch('security.service.get_settings') as mock_settings:
            settings_instance = MagicMock()
            settings_instance.defectdojo_url = ""
            settings_instance.defectdojo_api_key = ""
            mock_settings.return_value = settings_instance
            
            result = service.run_scan()
            
            print(f"[Service] Offline scan result status: {result.status}")
            self.assertEqual(result.status, "Completed (Offline)")
            self.assertGreater(result.total_packages, 0, "Should have counted packages in SBOM.")
            self.assertEqual(len(result.findings), 0, "Offline scan should have 0 findings.")

    @patch('requests.get')
    @patch('requests.post')
    def test_defectdojo_client_mock(self, mock_post, mock_get):
        """Mock the DefectDojo API server to test the client functions."""
        print("[Client] Testing DefectDojoClient API calls with mocked responses...")
        
        client = DefectDojoClient(
            url="http://mock-defectdojo:8080",
            api_key="mock_api_key",
            product_identifier="Test Product Name",
            engagement_identifier="Test Engagement Name",
        )

        # 1. Mock product ID lookup
        mock_get_product_response = MagicMock()
        mock_get_product_response.status_code = 200
        mock_get_product_response.json.return_value = {
            "results": [{"id": 42, "name": "Test Product Name"}]
        }
        
        # 2. Mock engagement lookup (not found)
        mock_get_engagement_response = MagicMock()
        mock_get_engagement_response.status_code = 200
        mock_get_engagement_response.json.return_value = {
            "results": []
        }
        
        # 3. Mock engagement creation
        mock_post_engagement_response = MagicMock()
        mock_post_engagement_response.status_code = 201
        mock_post_engagement_response.json.return_value = {
            "id": 99, "name": "Test Engagement Name"
        }

        # Setup mock side effects for requests
        def mock_get_side_effect(url, *args, **kwargs):
            if "products/" in url:
                return mock_get_product_response
            elif "engagements/" in url:
                return mock_get_engagement_response
            return MagicMock(status_code=404)

        mock_get.side_effect = mock_get_side_effect
        mock_post.return_value = mock_post_engagement_response

        # Execute ID resolution
        p_id = client.get_product_id()
        self.assertEqual(p_id, 42)

        e_id = client.get_engagement_id(product_id=p_id)
        self.assertEqual(e_id, 99)

        # 4. Mock SBOM import scan upload
        mock_upload_response = MagicMock()
        mock_upload_response.status_code = 201
        mock_upload_response.json.return_value = {
            "test": 101, "engagement": 99
        }
        mock_post.return_value = mock_upload_response

        # Create dummy file to allow upload_sbom to read it
        dummy_file = PROJECT_ROOT / "sbom" / "dummy_bom.json"
        dummy_file.parent.mkdir(parents=True, exist_ok=True)
        dummy_file.write_text("{}", encoding="utf-8")
        
        try:
            upload_res = client.upload_sbom(dummy_file, engagement_id=e_id)
            self.assertEqual(upload_res.get("test"), 101)
        finally:
            if dummy_file.exists():
                os.remove(dummy_file)

        # 5. Mock findings retrieval
        mock_findings_response = MagicMock()
        mock_findings_response.status_code = 200
        mock_findings_response.json.return_value = {
            "results": [
                {
                    "id": 501,
                    "component_name": "fastapi",
                    "component_version": "0.100.0",
                    "severity": "High",
                    "vulnerability_ids": [{"vulnerability_id": "CVE-2023-38039"}],
                    "description": "SQL injection in fastapi query handler.",
                    "mitigation": "Upgrade to fastapi >= 0.100.1",
                    "active": True
                }
            ],
            "next": None
        }
        
        def mock_get_side_effect_with_findings(url, *args, **kwargs):
            if "findings/" in url:
                return mock_findings_response
            return mock_get_side_effect(url, *args, **kwargs)
            
        mock_get.side_effect = mock_get_side_effect_with_findings
        
        findings = client.get_findings(test_id=101)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].package, "fastapi")
        self.assertEqual(findings[0].severity, "High")
        self.assertEqual(findings[0].cve, "CVE-2023-38039")
        print("[Client] Mocked API tests passed successfully!")


if __name__ == "__main__":
    unittest.main()
