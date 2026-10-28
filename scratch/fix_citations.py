from pathlib import Path

p = Path(r"C:\Users\gnana\OneDrive\Desktop\5thyear\MINI PROJECT\Project\final\Agentic-RAG\rag_engine.py")
lines = p.read_text(encoding="utf-8").splitlines()

new_lines = []
for line in lines:
    if 'norm = re.sub(r"(?i)\\bart\\.?\\s*", "Article ", art)' in line:
        continue
    elif 'norm = re.sub(r"(?i)\\bsec\\.?\\s*", "Section ", sec)' in line:
        continue
    elif 'found_citations.add(norm.title())' in line:
        continue
    elif 'for art in re.findall(r"\\b(?:Article|Art\\.?)\\s+\\d+' in line:
        new_lines.append('    for art in re.findall(r"\\b(?:Article|Art\\.?)\\s+(\\d+[A-Za-z]?(?:\\s*\\([0-9a-zA-Z]+\\))?)\\b", cleaned, re.IGNORECASE):')
        new_lines.append('        found_citations.add(f"Article {art}")')
    elif 'for sec in re.findall(r"\\b(?:Section|Sec\\.?)\\s+\\d+' in line:
        new_lines.append('    for sec in re.findall(r"\\b(?:Section|Sec\\.?)\\s+(\\d+[A-Za-z]?(?:\\s*\\([0-9a-zA-Z]+\\))*)\\b", cleaned, re.IGNORECASE):')
        new_lines.append('        found_citations.add(f"Section {sec}")')
    elif 'if not citation_found and top_prob < 0.60:' in line:
        new_lines.append('        if not citation_found:')
    else:
        new_lines.append(line)

p.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
print("Successfully patched rag_engine.py line-by-line!")
