FROM python:3.6-alpine

ADD agent.py . 

RUN pip3 install pygame numpy random matplotlib torch

CMD ["python", "agent.py"] 


