FROM tensorflow/tensorflow:1.9.0-gpu-py3

MAINTAINER ludong@cetc.com.cn

WORKDIR /app

COPY . .

RUN pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

CMD ["python", 'run.py']
