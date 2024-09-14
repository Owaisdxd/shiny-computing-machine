from ultralytics import YOLO
import smtplib
from email.message import EmailMessage
import os

def model_function(model):
    """
    训练YOLO模型。

    参数:
    model (YOLO): 要训练的YOLO模型实例。

    返回:
    bool: 如果训练成功，返回True；如果训练失败，返回False。
    """
    try:
        model.train(
            resume=False,
            data='my128.yaml',
            model='mydataa/images/last.pt',
            epochs=300,
            imgsz=1920,
            device='cuda:0,1,2,3,4,5,6,7,8,9',
            batch=30,
            workers=30,
            lr0=0.001,
            lrf=0.1,
            patience=20,
            optimizer='Adam',
            mosaic=False,
            mixup=False
        )
        return True
    except Exception as e:
        print(f"模型训练时出错: {e}")
        return False

def email_attachment(sender: str, receiver: str, passwd: str, cc: str, filename: str):
    """
    发送带有附件的电子邮件。

    参数:
    sender (str): 发送者的电子邮件地址。
    receiver (str): 接收者的电子邮件地址。
    passwd (str): 发送者电子邮件账户的密码。
    cc (str): 抄送的电子邮件地址。
    filename (str): 附件文件的路径。

    返回:
    None
    """
    try:
        # 设置SMTP服务器
        with smtplib.SMTP('smtp.gmail.com', 587) as s:
            s.starttls()
            s.login(user=sender, password=passwd)

            # 创建电子邮件消息
            msg = EmailMessage()
            msg['Subject'] = f'发送者 {sender} 附件 {filename} 发送给 接收者 {receiver}'
            msg['From'] = sender
            msg['To'] = receiver
            msg['Cc'] = cc
            msg.set_content("团队您好，附带媒体文件。")

            # 附加文件
            with open(filename, 'rb') as file:
                attach = file.read()
                file_name = file.name
                msg.add_attachment(attach, maintype='application', subtype='octet-stream', filename=file_name)

            # 发送电子邮件
            s.send_message(msg)
            print(f"从 {sender} 发送到 {receiver}，并抄送给 {cc}")

    except Exception as e:
        print(f"发送邮件时出错: {e}")

# 从环境变量中获取电子邮件凭据
sender = os.getenv('EMAIL_SENDER')
receiver = os.getenv('EMAIL_RECEIVER')
passwd = os.getenv('EMAIL_PASSWD')
cc = os.getenv('EMAIL_CC')
filename = "file1.html"

# 确保所有必要的环境变量都已设置
if not all([sender, receiver, passwd, cc]):
    print("请设置所有必需的环境变量: EMAIL_SENDER, EMAIL_RECEIVER, EMAIL_PASSWD, EMAIL_CC")
else:
    # 初始化并训练模型
    model = YOLO('yolov8m')
    training_successful = model_function(model)

    # 如果训练失败则发送电子邮件
    if not training_successful:
        email_attachment(sender, receiver, passwd, cc, filename)
