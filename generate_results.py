import math
import os
import json

def read_data(method, dataset):
    res_list = os.listdir(f"result/{dataset}")
    res_list = [res for res in res_list if res.startswith(method)]
    if not res_list:
        return []   
    res_list = [res.split("_")[-1] for res in res_list] 
    res_list = [float(res.split(".json")[0]) for res in res_list]
    return res_list

def compute_mean_std(res_list):
    if not res_list:
        return 0, 0
    mean = sum(res_list) / len(res_list)
    std = math.sqrt(sum([(res - mean)**2 for res in res_list]) / len(res_list))
    return mean, std

if __name__ == "__main__":
    dataset_name = "CoLA"
    res_dict = {}
    for method in ["ilra", "lora", "bit_fit", "krona", "fft", "pa"]:
        res = read_data(method, dataset_name)
        res = compute_mean_std(res)
        res_dict[method] = {
            "mean": res[0],
            "std": res[1]   
        }
    
    with open(f"result/res_for_plot/{dataset_name}.json", "w") as f:
        json.dump(res_dict, f, indent=4 ,ensure_ascii=False)
    