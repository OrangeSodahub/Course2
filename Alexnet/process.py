import os
import cv2
import random
import shutil
import translate

# F:\PersonalFiles\CourseMaterials\大二下\机器学习\Animals-10

# resize
def image_tailor(input_dir, out_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            # file为root目录中的文件
            filepath = os.path.join(root, file)  # 连接两个或更多的路径名组件，filepath路径为/root/file
            
            try:
                image = cv2.imread(filepath)  # 根据输入路径读取照片
                dim = (227, 227)  # 裁剪的尺寸
                resized = cv2.resize(image, dim)  # 按比例将原图缩放成227*227
                path = os.path.join(out_dir, file)  # 保存的路径和相应的文件名
                cv2.imwrite(path, resized)  # 进行保存
            except:
                print(filepath)
                os.remove(filepath)
        cv2.waitKey()


# rename
def image_rename(input_dir,name_start):
    count = 1
    filelist = os.listdir(input_dir)

    for file in filelist:
        old_file = os.path.join(input_dir,file)

        if os.path.isfile(old_file):
            new_file = os.path.join(input_dir, name_start + '_' + str(count) + '.jpg')
            os.rename(old_file, new_file)
            count += 1
        else:
            continue
        pass
    print(str(count - 1) + " files have created.")


# merge
def files_merge(old_path, new_path):
    filenames = os.listdir(old_path)
    target_path = new_path
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    for file in filenames:
        sonDir = old_path + file
        for root, dirs, files in os.walk(sonDir):
            if len(files) > 0:
                for f in files:
                    newDir = sonDir + '/' + f
                    shutil.copy(newDir, target_path)
            else:
                print(sonDir + "is empty.")


# split train and test dataset
def move_image(dirpath, destdir):
    fs = os.listdir(dirpath)
    random.shuffle(fs)     # 随机将文件目录整理
    le = int(len(fs)*0.2)

    for f in fs[0:le]:
        file_path = dirpath + "/" + f
        shutil.move(file_path, destdir)

if __name__ == '__main__':
    # preprocess
    # rename
    # for i in range(10):
    #     in_dir = os.path.join("F:/PersonalFiles/CodeWorks/Python/AlexNet/Course2/Animals-10/raw-img/",translate.num[i])
    #     image_rename(in_dir, translate.translate[translate.num[i]])

    # merge
    # files_merge("F:F:/PersonalFiles/CodeWorks/Python/AlexNet/Course2/Animals-10/raw-img/","F:F:/PersonalFiles/CodeWorks/Python/AlexNet/Course2/Animals-10/train/")

    # split
    # dirpath = r"F:/PersonalFiles/CodeWorks/Python/AlexNet/Course2/Animals-10/train"
    # destdir = r"F:/PersonalFiles/CodeWorks/Python/AlexNet/Course2/Animals-10/test"
    # move_image(dirpath, destdir)

    # resize
    input_dir = r"F:/PersonalFiles/CodeWorks/Python/AlexNet/Course2/Animals-10/train/"
    out_dir = r"F:/PersonalFiles/CodeWorks/Python/AlexNet/Course2/Animals-10/resize_train"
    image_tailor(input_dir, out_dir)
    pass