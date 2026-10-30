# 窑智 · 部署专用仓库

本仓库仅用于 **自动部署魔搭创空间** 的 CI/CD，只含 `space/` 与 `yaozhi/` 运行代码，不含参赛材料与个人信息。

- `deploy.yml`：push 到 main 后同步 `space/`+`yaozhi/` 至魔搭空间并触发部署
- `keepalive.yml`：定时保活
