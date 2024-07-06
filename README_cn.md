## 部署

1. 从git拉取项目

   ```bash
   git clone http://gitea.fcunb.cn:10083/Echo/label-studio-ml-backend.git
   ```

2. mydata目录授权

   ```bash
   cd label-studio-ml-backend
   sudo chown -R 1001:1001 mydata/
   ```

3. 使用docker部署

   ```bash
   docker-compose up -d
   ```

   