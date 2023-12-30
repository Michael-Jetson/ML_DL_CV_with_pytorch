
import json

if __name__=="__main__":




# Step 1: 读取JSON文件
    with open(r"C:\Users\pengf\Documents\WildRefer_original_submit_new_del_38.json",'r') as file:  
        num=0  
        datas = json.load(file)
        print(type(datas))
        print(datas[0])
        print(datas[1])
        
        for idx,data in enumerate(datas):
            if len(data['token'])>36:
                print(len(data['token']))
                datas.pop(idx)
            if data['data']=='09-23-4':
                datas[idx]['data']='09-23-13-44-53-4'
            if data['data']=='09-23-3':
                datas[idx]['data']='09-23-13-44-53-3'
            if data['data']=='09-23-2':
                datas[idx]['data']='09-23-13-44-53-2'
    # Step 3: 保存修改后的数据到新的JSON文件
    with open(r'C:\Users\pengf\Documents\WildRefer_original_submit_new_del_36.json','w') as file:
        json.dump(datas, file, indent=4)  # indent参数可使输出的JSON文件格式更美观


