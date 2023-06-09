import json

def get_data_from_jsonl(path):
    data = []
    with open(path, 'r',  encoding='utf-8') as file:
        for line in file:
            item = json.loads(line)
            if len(item['positive_case_methods'])==0:
                continue
            methods_no_of_lines = []
            for sample in item['positive_case_methods']:
                methods_no_of_lines.append(count_lines(sample))
            data.append(calc_average(methods_no_of_lines))
    return data

def generate_posvsneg_ratio(path, loc_threshold):
    data = {}
    total_count = 0
    zero_pos_count = 0
    above_total = 0
    below_total =0
    with open(path, 'r',  encoding='utf-8') as file:
        for line in file:
            item = json.loads(line)
            if len(item['positive_case_methods'])==0:
                zero_pos_count +=1
                continue
            metrics = {
                    'above_threshold_count': 0,
                    'below_threshold_count': 0,
                    'ratio': 0.0
                }
            for sample in item['positive_case_methods']:
                total_count += 1
                if count_lines(sample) >= loc_threshold:
                    metrics['above_threshold_count'] = metrics.get('above_threshold_count')+1
                else:
                    metrics['below_threshold_count'] = metrics.get('below_threshold_count')+1
                
            metrics['ratio'] = ratio(metrics['above_threshold_count'], metrics['below_threshold_count'])
            above_total += metrics['above_threshold_count']
            below_total += metrics['below_threshold_count']
            data[item['repo_name']] = metrics
    return data, above_total, below_total, total_count, zero_pos_count

def ratio(a, b):
    a = float(a)
    b = float(b)
    if b == 0:
        return a
    return round(a/b, 5)

def count_lines(snippet: str):
    lines = snippet.split('\n')
    total_lines = len(lines)
    return total_lines

def calc_average(list: list):
    total = sum(list)
    num_elements = len(list)
    average = total / num_elements
    return int(average)

FILE_PATH = r'D:\DevHub\extract-method-identification\file_0000.jsonl'

if __name__=="__main__":
    
    avg_method_loc_data = get_data_from_jsonl(FILE_PATH)
    avg_loc_threshold = calc_average(avg_method_loc_data)
    ratio_map, a, b, total_count, zero_pos_count  = generate_posvsneg_ratio(FILE_PATH, avg_loc_threshold)
    # TODO: Handle according to the need
    print(a/b)
    print(total_count)
    print(zero_pos_count)