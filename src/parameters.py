
DT = 0.1  

# a,b,c,d are  model paramters  , v_init is  model starting V , and T is the duration of simulation in ms 

PATTERNS = {
    "RS":dict(
        name ="regular spiking",
        a=0.025, b=0.2 , c=-65 , d=8 , v_init=-65 ,
        T=200, schedule=[(0, 0),(20, 10)]),

    "IB": dict(
        name = "intrinsically bursting",
        a=0.0225, b=0.2, c=-55 , d=4 , v_init=-65,
        T=200, schedule=[(0, 0),(20, 10)]),

    "CH": dict(
        name ="chattering",
        a=0.02, b=0.2, c=-50, d=2, v_init=-65,
        T=200, schedule=[(0, 0), (20, 10)]),

    "FS": dict(
        name ="fast spiking",
        a=0.1, b=0.2 , c=-65 , d=2 ,  v_init=-65 ,
        T=200, schedule=[(0, 0), (20, 10)]),

    "LTS": dict(
        name="low-threshold spiking",
        a=0.02, b=0.25, c=-65, d=2, v_init=-65,
        T=200, schedule=[(0, 0), (20, 10)]),

    "TC_tonic": dict(
        name="thalamo-cortical (tonic)",
        a=0.02, b=0.25, c=-65, d=0.05, v_init=-64.41,
        T=200, schedule=[(0, 0), (20, 1.5)]),

    
    "TC_rebound": dict(
        name ="thalamo-cortical (rebound burst)",
        a=0.02, b=0.25, c=-65, d=0.05, v_init=-87,
        T=300, schedule=[(0, -10), (200, 0)]),

    "RZ": dict(
        name="resonator",
        a=0.1, b=0.26, c=-65, d=2, v_init=-62.5,
        T=200, schedule=[(0, 0), (10, 0.2), (97, 1), (102, 0.2)]),

}

#  RS with low / medium / high step currents
STEP_LEVELS = {"low": 4, "medium": 10, "high": 40}

STEP_PATTERNS = {
    f"RS_{level}": {**PATTERNS["RS"],
                    "name": f"regular spiking, I={I} ({level})",
                    "T": 300,
                    "schedule": [(0, 0), (20, I)]}
    for level, I in STEP_LEVELS.items()
}