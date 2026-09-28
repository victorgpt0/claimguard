# notebooks/

Colab experimentation only — nothing here is the source of truth. Once a notebook cell
works, promote the logic into `src/` as a proper module and import it, rather than letting
the notebook itself become the implementation. See `implementation_roadmap.md` Section 2
for the standard Colab bootstrap cell (mount Drive, clone/pull repo, install
requirements, add `src/` to path) — paste it at the top of every notebook here.
