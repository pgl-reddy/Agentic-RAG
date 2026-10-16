import shutil
import base64
from pathlib import Path

src = Path(r"C:\Users\Vybha\.gemini\antigravity-ide\brain\26493043-ee59-4102-beff-3c715ce5832a\media__1784889718562.png")
dst = Path(r"c:\Users\Vybha\projects\Agentic-RAG\legal_hero.png")

shutil.copy(src, dst)
print("Copied image to", dst)

with open(dst, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")

print(f"Base64 length: {len(b64)}")
with open(Path(r"c:\Users\Vybha\projects\Agentic-RAG\legal_hero_b64.txt"), "w") as f:
    f.write(b64)
