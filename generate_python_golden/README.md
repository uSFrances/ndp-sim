# Generate Python Golden

这是一个用于一键生成模型输入、按配置裁剪权重及生成 `python_golden` 数据的自动化工具。

## 依赖
- Python 3
- numpy
- tqdm

可以通过以下命令安装依赖：
```bash
pip install numpy tqdm
```

## 配置
核心参数在 `config.json` 中定义。调整这些参数会影响输入维度以及模型推理时的形状。
额外支持 `"target_op"` 参数用于指定需要提取局部数据的模块（比如 `"rms_norm"`），相应的代码须放在 `single_op_data/` 文件夹中。

## 如何让别人使用此仓库
其他人若要使用此工具，只需在终端依次执行以下命令：

```bash
# 1. 下载仓库代码到本地
git clone https://github.com/JasmineLuuu/generate_python_golden.git

# 2. 进入项目目录
cd generate_python_golden

# 3. 安装依赖（如果尚未安装）
pip install numpy tqdm

# 4. ⚠️ 将原模型权重文件夹（如 DeepSeek-R1-Distill-Qwen-1.5B-f16）放入该目录中。
# 5. 一键生成数据
make
```

## 使用方法

本仓库提供了一个 `Makefile` 来简化执行流程。打开终端，进入本目录，输入以下命令即可：

- **一键执行完整流程 (推荐)**
  ```bash
  make
  ```
  该命令会依次执行：生成虚拟输入 -> 生成/裁剪权重 -> 生成 python_golden 数据。

- **清理生成的数据**
  ```bash
  make clean
  ```

- **查看所有可用命令**
  ```bash
  make help
  ```

## 目录及流程说明
1. `config.json`: 存储模型与序列的参数。
2. `create_dummy_inputs.py`: 根据 `config.json` 中的 `sequence_length` 等参数生成虚拟的 `.bin` 输入存放于 `inputs` 文件夹。
3. `weight_gen.py`: 读取模型参数并切割生成对应的权重保存于 `model_weights_small` 文件夹。
4. `deepseek1.5b_3_time_golden_smallsize.py`: 主脚本，运行并产生追踪数据于 `python_golden` 文件夹中。
5. `run_single_op.py`: 根据 `config.json` 中的 `target_op` 指令，调用 `single_op_data/<target_op>.py` 脚本处理切片，生成对应的 `op.json`。

## 开发者维护指南（如何上传到 GitHub）

### 情景 1：日常修改了文件后的更新（最常用）
如果您修改了本地的 `.py`、`.json` 或其他文件，只需执行以下三步即可同步到 GitHub：

```bash
# 1. 保存所有改动
git add .

# 2. 提交此次修改（引号内可以写明您改了什么，例如 "修改了 config.json 的序列长度"）
git commit -m "Update files"

# 3. 推送到 GitHub
git push
```

### 🤝 如何邀请其他人一起编辑仓库？
如果您希望您的同学或同事也能直接向这个仓库提交（Push）修改，您需要将他们添加为协作者：
1. 在电脑浏览器中打开您的 GitHub 仓库页面 (`https://github.com/JasmineLuuu/generate_python_golden`)。
2. 点击页面上方导航栏右侧的 ⚙️ **Settings** (设置)。
3. 在左侧菜单栏中找到 **Collaborators** (协作者) 并点击。
4. 点击绿色的 **Add people** (添加人员) 按钮。
5. 输入对方的 GitHub 用户名或注册邮箱，并在下拉列表中选中他们。
6. 点击按钮确认添加。
7. **注意**：对方会收到一封邀请邮件（也可以在他们的 GitHub 网页顶部看到气泡通知）。**对方必须点击接受邀请**，之后他们也能通过 `git push` 直接更新这个仓库的代码了。

### 🔄 如何与别人的仓库合并代码？ (Pull Request 机制)
如果您没有把对方加为协作者，或者您想给开源社区的其他项目贡献代码，可以通过以下标准的 GitHub 流程来实现代码合并：

**情况 A：你想把修改合并到别人的仓库**
1. **Fork（派生）**：在别人的 GitHub 仓库页面右上角，点击 **Fork** 按钮，这会在您的账号下复制一份该仓库。
2. **Clone（克隆）**：将您自己账号下的这份代码 `git clone` 到本地电脑。
3. **修改并 Push**：在本地修改代码后，按照情景 1 的方法 `add`, `commit`, `push` 到您自己的仓库中。
4. **Pull Request（拉取请求）**：回到 GitHub 页面，点击 **Contribute** -> **Open pull request**。填写说明后提交。原作者收到请求并同意后，您的代码就会被合并到他们的仓库中！

**情况 B：别人想把修改合并到你的仓库**
1. 对方 Fork 您的仓库并修改代码，然后向您提交 Pull Request。
2. 您会在您的仓库页面的 **Pull requests** 标签签里看到一个新请求。
3. 点击进去审核对方修改的代码，确认无误后点击绿色的 **Merge pull request** 按钮，对方的代码就会完美融合进您的 `main` 分支了。

### 🔀 如何将两个完全独立的仓库合并成一个？
如果您和别人分别建了并开发了两个没有任何交集的仓库，现在想要把它们合并到一个里，请在您的本地终端这样操作：

```bash
# 1. 确保您在自己的这个项目目录下
cd generate_python_golden

# 2. 将别人的仓库作为一个新的数据源添加进来（取个代号，比如 other-repo）
git remote add other-repo <别人的仓库URL>

# 3. 将别人仓库的代码和历史记录拉取到本地
git fetch other-repo

# 4. 强制合并别人仓库的 main 分支到你当前的分支（核心步骤）
git merge other-repo/main --allow-unrelated-histories

# 5. （可选）如果你和别人修改了同一个文件，这里可能会提示代码冲突(Conflict)。
#            用编辑器打开冲突的文件，手动决定保留谁的代码，然后保存：
#            git add .
#            git commit -m "Resolve merge conflicts"

# 6. 将合并后的最终代码推送到你自己的 GitHub 仓库
git push origin main
```

### 情景 2：第一次上传代码（初始化新仓库）
如果您是第一次将这个文件夹推送到 GitHub，请按照以下步骤操作：

**1. 在 GitHub 创建空仓库：**
- 登录 GitHub，点击右上角 `+` -> "New repository"。
- 填写 Repository name，**不要** 勾选初始化 README/.gitignore/License。
- 点击 "Create repository"，复制生成的仓库 URL (`<your-repo-url>`).

**2. 在本地终端执行推送：**
```bash
# 初始化 git（如果尚未初始化）
git init

# 添加更改到暂存区并提交
git add .
git commit -m "Update python golden generator"

# 关联远程仓库并推送（使用 HTTPS 链接更简单，无需配置 SSH 密钥）
git remote add origin https://github.com/JasmineLuuu/generate_python_golden.git
git branch -M main
git push -u origin main
```

**⚠️ 常见提示与报错:**
- **需要密码登录:** 首次推送时，如果系统弹窗或终端提示输入账号密码，请按提示登录您的 GitHub 账号（注意，如果在终端中输入密码，需要使用 GitHub 的 Personal Access Token 作为密码）。
- **`error: remote origin already exists.`:** 如果出现此报错，说明之前已经关联过别的地址，请将 `git remote add ...` 命令替换为 `git remote set-url origin https://github.com/JasmineLuuu/generate_python_golden.git` 然后再次 push 即可。
- **`error: RPC failed; HTTP 408 / 413` 或卡在 Writing objects 的超大文件:** 
  如果在 `.gitignore` 生效前不小心 add/commit 了原模型文件夹（好几个GB），导致被记录在 Git 历史中无法推送，对于新仓库，推荐直接重置本地 Git 初始化：
  ```bash
  rm -rf .git
  git init
  git add .
  git commit -m "clean commit"
  git remote add origin https://github.com/JasmineLuuu/generate_python_golden.git
  git branch -M main
  git push -u origin main
  ```

### 如何确认是否成功上传到了 GitHub？
如果推送成功，回到您的 GitHub 仓库网页并刷新，原本的 "Quick setup" 指南页面应该会消失，直接显示文件夹中的各种代码文件（如 `.py`，`config.json` 等）和这份 `README.md` 文档。如果仍然显示 "Quick setup"，则说明代码尚未推送成功，请重试 `git push -u origin main`。
