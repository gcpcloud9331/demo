**Steps**

After cloning the repository run the following commands,

docker build -t app:v1 .

docker run -d --name pytorch-app -p 8080:8080 app:v1
