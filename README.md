# finetuning_esm2_public
Setting up generalizable repository to finetune esm2 models
---
#### Files
finetuning_esm2.py --> Python script to finetune esm2
ESM2_w_regression_MLP_heads.py --> dataloader and model architexture loaded in finetuning_esm2.py
functions.py --> contains loaded functions important for finetuning esm2
datasets --> csv and pkl files without split in the filename
pkl files with splits in the filename --> these are fixed dataplits for datasets to avoid test leakage

#### Create SSH key in Setting on GitHub

1. **Generate new key**: 
    ```bash
    ssh-keygen -t ed25519 -C "[your_email@example.com]" -f "/Users/nathanielblalock/.ssh/id_ed25519_new"
    ```
  
2. **Make sure SSH is running**: 
    ```bash
    eval "$(ssh-agent -s)"
    ```

3. **Add your new SSH key to the SSH agent**: 
    ```bash
    ssh-add ~/.ssh/id_ed25519_new
    ```

4. **Copy the new SSH public key to your clipboard**: 
    ```bash
    pbcopy < ~/.ssh/id_ed25519_new.pub
    ```

5. **Go to GitHub and navigate to Settings > SSH and GPG keys**.

6. **Click on New SSH key to add the new key**. 
    - Give it a title and paste the key into the "Key" field. 
    - Then save it.

7. **Configure SSH to use the correct SSH key** by editing the `.ssh/config` file to include:

    ```text
    # Use the new key for GitHub
    Host github.com
    HostName github.com
    User username@wisc.edu
    IdentityFile ~/.ssh/id_ed25519_new
    ```

---

#### Cloning the Repository

1. **Open a terminal and navigate to your desktop (or the folder where you want the repository to be located):**

    ```bash
    cd ~/Desktop
    ```

2. **Clone the repository:**

    ```bash
    git clone git@github.com:RomeroLab/RLXF_Projects.git
    ```

---

#### Pushing Edits to GitHub

1. **Navigate to the repository directory:**

    ```bash
    cd RLXF_Projects
    ```

2. **Check the status of your local repository:**

    ```bash
    git status
    ```

3. **Add your changes to the staging area:**

    ```bash
    git add .
    ```

    Use `.` to add all changed files, or specify the individual files.

4. **Commit your changes:**

    ```bash
    git commit -m "Your commit message"
    ```

5. **Push the changes to GitHub:**

    ```bash
    git push origin main
    ```

    Replace `main` with the branch you're working on, if different.

---

#### Pulling Updates from GitHub

1. **Ensure you're in the repository directory:**

    ```bash
    cd RLXF_Projects
    ```

2. **Pull the latest changes:**

    ```bash
    git pull origin main
    ```

    Replace `main` with the branch you're working on, if different.


