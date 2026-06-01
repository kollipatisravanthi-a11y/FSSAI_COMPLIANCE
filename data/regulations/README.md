Place your official regulations corpus here (preferred: `.txt` exports).

Recommended sources (you supply the documents):
- FSSAI Food Safety and Standards Act, 2006
- FSSAI Schedule 4 (GHP/GMP)
- HACCP principles and related guidelines

Then run:

```powershell
python .\scripts\build_vectorstore.py
```

This will create a local ChromaDB index in `chroma/`.
