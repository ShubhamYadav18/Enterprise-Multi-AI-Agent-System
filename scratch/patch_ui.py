import sys

file_path = r"d:\MULTI AGENT\enterprise_multi_agent\ui\app.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# The placeholder inserted by indent_ui.py:
placeholder = """elif view_mode == 'Security':
    from security.service import SecurityService
    st.markdown(\"\"\"
    <div class=\"main-header\">
        <h1>🛡️ Security & Software Supply Chain</h1>
        <p>SBOM Generation • Vulnerability Scanning • DefectDojo Integration</p>
    </div>
    \"\"\", unsafe_allow_html=True)

    st.info('Security Dashboard is under construction.')
"""

# Replace with function call
replacement = """elif view_mode == 'Security':
    render_security_dashboard()
"""
if placeholder in content:
    content = content.replace(placeholder, replacement)

# Add the function definition at the bottom
with open(r"d:\MULTI AGENT\enterprise_multi_agent\scratch\security_dashboard.py", "r", encoding="utf-8") as dash:
    dash_content = dash.read()

content += "\n\n"
content += dash_content

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
