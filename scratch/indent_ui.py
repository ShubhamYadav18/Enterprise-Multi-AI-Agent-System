import sys

file_path = r"d:\MULTI AGENT\enterprise_multi_agent\ui\app.py"
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    if i == 255: # zero-indexed, line 256
        new_lines.append("if view_mode == 'AI Control Tower':\n")
    if i >= 255:
        if line.strip() == "":
            new_lines.append(line)
        else:
            new_lines.append("    " + line)
    else:
        new_lines.append(line)

new_lines.append("\n")
new_lines.append("elif view_mode == 'Security':\n")
new_lines.append("    from security.service import SecurityService\n")
new_lines.append("    st.markdown(\"\"\"\n")
new_lines.append("    <div class=\\\"main-header\\\">\n")
new_lines.append("        <h1>🛡️ Security & Software Supply Chain</h1>\n")
new_lines.append("        <p>SBOM Generation • Vulnerability Scanning • DefectDojo Integration</p>\n")
new_lines.append("    </div>\n")
new_lines.append("    \"\"\", unsafe_allow_html=True)\n")
new_lines.append("\n")
new_lines.append("    st.info('Security Dashboard is under construction.')\n") # I will replace this with the real code later

with open(file_path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)
