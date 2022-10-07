
import os
import cv2
import torch
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")
                                                 
def getframes():
     
    image_1_path = r'C:\Users\siliconarts\Desktop\siliconarts_study\team_leader_targer_paper\rife_c_python\practice_6\rife_c_python\x64\Debug\demo\I0_0.png'
    image_2_path = r'C:\Users\siliconarts\Desktop\siliconarts_study\team_leader_targer_paper\rife_c_python\practice_6\rife_c_python\x64\Debug\demo\I0_1.png'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
       
    from train_log.RIFE_HDv3 import Model
    model = Model()  
    model.load_model('train_log', -1)   
    model.eval()
    model.device()
    print("Load model successfully")

    img0 = cv2.imread(image_1_path, cv2.IMREAD_UNCHANGED)
    img1 = cv2.imread(image_2_path, cv2.IMREAD_UNCHANGED)
    img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

    n, c, h, w = img0.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)
    img0 = F.pad(img0, padding)
    img1 = F.pad(img1, padding)
                                  
    img_list = [img0, img1]
    for i in range(4):
        tmp = []
        for j in range(len(img_list) - 1):
            mid = model.inference(img_list[j], img_list[j + 1])
            tmp.append(img_list[j])
            tmp.append(mid)
        tmp.append(img1)
        img_list = tmp

    if not os.path.exists('output'):
        os.mkdir('output')
    
    print("Save interpolation frames")  
    for i in range(len(img_list)):
        cv2.imwrite('C:/Users/siliconarts/Desktop/siliconarts_study/team_leader_targer_paper/rife_c_python/practice_6/rife_c_python/x64/Debug/output/img{}.png'.format(i), (img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])
    print("Save done")    
    
    lst = os.listdir("C:/Users/siliconarts/Desktop/siliconarts_study/team_leader_targer_paper/rife_c_python/practice_6/rife_c_python/x64/Debug/output")
    number_files = len(lst)
       
    print("Display interpolation frames")
    for i in range (number_files):
        image_path_i = cv2.imread("C:/Users/siliconarts/Desktop/siliconarts_study/team_leader_targer_paper/rife_c_python/practice_6/rife_c_python/x64/Debug/output/img" + str(i) + ".png")
        window_name = 'img'+ str(i) + ".png"
        cv2.imshow(window_name, image_path_i)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()

    print("Display done")
               
                                                    

if __name__ == "__main__":
    getframes()                                                                  