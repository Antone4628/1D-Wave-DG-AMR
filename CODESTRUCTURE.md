1D_WAVE_AMR/
├── numerical/
    ├── __init__.py
    ├── amr/
    │   ├── __init__.py
    │   ├── adapt.py
    │   ├── forest.py
    │   └── projection.py
    ├── dg/
    │   ├── __init__.py
    │   ├── basis.py
    │   └── matrices.py
    ├── grid/
    │   ├── __init__.py
    │   └── mesh.py
    └── solvers/
        ├── __init__.py
        ├── dg_wave_solver.py  # New class file
        ├── wave.py           # Your existing wave solver
        └── utils.py