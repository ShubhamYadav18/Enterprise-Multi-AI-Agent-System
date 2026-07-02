import streamlit as st
import time
from security.service import SecurityService
from security.sbom import generate_sbom

def render_security_dashboard():
    from security.service import SecurityService
    
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ Security & Software Supply Chain</h1>
        <p>SBOM Generation • Vulnerability Scanning • DefectDojo Integration</p>
    </div>
    """, unsafe_allow_html=True)
    
    service = SecurityService()
    
    # ---------------------------------------------------------
    # Actions
    # ---------------------------------------------------------
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📦 Generate SBOM")
        st.caption("Generate a CycloneDX SBOM using cdxgen.")
        if st.button("Generate SBOM", use_container_width=True):
            with st.spinner("Generating SBOM..."):
                res = generate_sbom()
                if res.status == "success":
                    st.success("SBOM Generated Successfully!")
                    st.markdown(f"**Time:** {res.timestamp}")
                    st.markdown(f"**Output Path:** `{res.output_file}`")
                else:
                    st.error(f"Generation Failed: {res.error}")

    with col2:
        st.markdown("### ☁️ Upload & Scan")
        st.caption("Upload SBOM to DefectDojo and retrieve findings.")
        if st.button("Run Full Security Scan", use_container_width=True):
            with st.spinner("Running security scan (this may take a moment)..."):
                result = service.run_scan()
                if result.status == "Completed":
                    st.success("Scan Completed!")
                    st.markdown(f"**Scan ID:** {result.scan_id}")
                else:
                    st.error(f"Scan failed: {result.error}")
    
    st.markdown("---")
    
    # ---------------------------------------------------------
    # Dashboard Display
    # ---------------------------------------------------------
    latest = service.get_latest_scan_result()
    if not latest:
        st.info("No scans have been run yet. Generate an SBOM and run a scan to see results here.")
        return
        
    st.markdown("## 📊 Security Summary")
    st.caption(f"Latest Scan ID: `{latest.scan_id}` (Time: {latest.timestamp})")
    
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total Packages", latest.total_packages)
    m2.metric("Critical", latest.critical_count)
    m3.metric("High", latest.high_count)
    m4.metric("Medium", latest.medium_count)
    m5.metric("Low", latest.low_count)
    m6.metric("Informational", latest.info_count)
    
    st.markdown("---")
    
    # Tabs for Vulnerabilities, Dependencies, History
    tab1, tab2, tab3 = st.tabs(["🐛 Vulnerabilities", "🌳 Dependency Tree", "📜 Scan History"])
    
    with tab1:
        st.markdown("### Vulnerability Findings")
        if not latest.findings:
            st.success("No vulnerabilities found in the latest scan!")
        else:
            # Convert to DataFrame for easy Streamlit table
            import pandas as pd
            data = []
            for f in latest.findings:
                data.append({
                    "Package": f.component_name,
                    "Version": f.component_version,
                    "Severity": f.severity,
                    "CVE": f.cve,
                    "Status": f.status,
                    "Description": f.description[:100] + "..." if len(f.description) > 100 else f.description
                })
            df = pd.DataFrame(data)
            
            # Simple severity sort trick
            sev_map = {"Critical": 1, "High": 2, "Medium": 3, "Low": 4, "Info": 5, "Informational": 5}
            df["_sev_sort"] = df["Severity"].map(sev_map)
            df = df.sort_values("_sev_sort").drop(columns=["_sev_sort"])
            
            st.dataframe(df, use_container_width=True)
            
    with tab2:
        st.markdown("### Dependency Tree")
        try:
            tree = service.get_dependency_tree()
            st.json(tree)
        except Exception as e:
            st.warning("Dependency tree not available. Run SBOM generation first.")
            
    with tab3:
        st.markdown("### Scan History")
        history = service.get_history()
        if not history:
            st.info("No history available.")
        else:
            import pandas as pd
            h_data = [h.model_dump() for h in history]
            st.dataframe(pd.DataFrame(h_data), use_container_width=True)
    
    st.markdown("---")
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        sbom_path = service.project_root / "sbom" / "bom.json"
        if sbom_path.exists():
            with open(sbom_path, "rb") as f:
                st.download_button("📥 Download SBOM JSON", data=f, file_name="bom.json", mime="application/json")
    with d_col2:
        report_data = latest.model_dump_json(indent=2)
        st.download_button("📥 Download Scan Report", data=report_data, file_name=f"scan_{latest.scan_id}.json", mime="application/json")
