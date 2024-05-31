import math

def estimate_time(input_json, peak_double, peak_float):
    def gemm_flops(key):
        return 2*key["m"]*key["k"]*key["n"]
    
    def syrk_flops(key):
        return 2*key["k"]*key["n"]*(key["n"]+1)
    
    def trsm_flops(key):
        if key["side"] == "Left":
            return key["n"]*key["m"]*key["m"]
        else:
            return key["n"]*key["n"]*key["m"]
    
    calc_flops = {}
    calc_flops["gemm"] = gemm_flops
    calc_flops["syrk"] = syrk_flops
    calc_flops["trsm"] = trsm_flops
    
    method = input_json["keywords"]["method"]
    
    multiplier = 1 
    if input_json["keywords"]["run_type"] == "exhaustive":
        if method == "gemm":
            multiplier = 8
        if method == "syrk":
            multiplier = 4
        if method == "trsm":
            multiplier = 4
    
    repetitions = int(input_json["keywords"]["repetitions"])
    redundancy = 10
    multiplier *= repetitions*redundancy
    
    data_type = input_json["keywords"]["data_type"]
    if data_type == "double":
        multiplier /= peak_double
    elif data_type == "float":
        multiplier /= peak_float
    else:
        raise Exception(f"Invalid data type {data_type}")
    
    
    
    problems = input_json["problems"]
    seconds = 0.0
    for problem in problems:
        seconds += multiplier*calc_flops[method](problem)/1024.0**4
    
    hours=math.ceil(seconds/3600)
    return hours

