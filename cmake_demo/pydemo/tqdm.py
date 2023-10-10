import tqdm
import json
if __name__=="__main__":


# Step 1: 读取JSON文件
    with open(r"C:\Users\pengf\Documents\WildRefer_original_submit.json",'r') as file:    
        datas = json.load(file)
        print(datas[0])
        for data in datas:
            if data['data']=="09-23-3":
                data['data']="09-23-13-44-53-3"


    # Step 3: 保存修改后的数据到新的JSON文件
    with open(r'C:\Users\pengf\Documents\WildRefer_original_submit_new.json','w') as file:
        json.dump(datas, file, indent=4)  # indent参数可使输出的JSON文件格式更美观


