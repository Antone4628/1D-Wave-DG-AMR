1D_WAVE_AMR/
├── numerical/
│   ├── __init__.py
│   ├── amr/
│   │   ├── __init__.py
│   │   ├── adapt.py
│   │   ├── forest.py
│   │   └── projection.py
│   ├── dg/
│   │   ├── __init__.py
│   │   ├── basis.py
│   │   └── matrices.py
│   ├── grid/
│   │   ├── __init__.py
│   │   └── mesh.py
│   ├── solvers/
│   │   ├── __init__.py
│   │   ├── dg_wave_solver.py  
│   │   ├── wave.py           
│   │   └── utils.py
│   └── environments/
│       ├── __init__.py
│       └── dg_amr_env.py
├── scripts/
│    ├── 1D_wave_amr.py
│    ├── forest_example.py
│    ├── snippets.py
│    └── train_amr.py
├── tests/
│   ├── __init__.py
│   ├── test_amr/
│   │   ├── __init__.py
│   │   ├── balance_test.py
│   │   ├── class_test.py
│   │   └── projection_test.py
│   ├── test_animation/
│   │   ├── __init__.py
│   │   └── manim_test.py
│   ├── test_RL/
│   │   ├── __init__.py
│   │   ├── check_env.py
│   │   ├── test_env.py
│   │   └── train_test.py
├── CODESTRUCTURE.md
├── README.md
├── requirements.txt
└── setup.py
