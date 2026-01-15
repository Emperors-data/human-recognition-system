# Step-by-Step Guide: Upload to GitHub

## Prerequisites
- GitHub account
- Git installed on your computer

## Step 1: Install Git (if not already installed)

Download and install Git from: https://git-scm.com/download/win

## Step 2: Configure Git (First Time Only)

Open PowerShell in your project directory and run:

```powershell
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Step 3: Create GitHub Repository

1. Go to https://github.com
2. Click the **"+"** icon (top right) → **"New repository"**
3. Fill in:
   - **Repository name**: `human-recognition-system`
   - **Description**: "Real-time human recognition and activity understanding system"
   - **Visibility**: Public or Private (your choice)
   - **DO NOT** check "Initialize with README" (we already have one)
4. Click **"Create repository"**

## Step 4: Initialize Git in Your Project

In PowerShell, navigate to your project and run:

```powershell
cd "d:\Mini Project"
git init
```

## Step 5: Create .gitignore File

Create a `.gitignore` file to exclude unnecessary files:

```powershell
# This is already created for you - just verify it exists
```

## Step 6: Add Files to Git

```powershell
git add .
```

## Step 7: Commit Your Changes

```powershell
git commit -m "Initial commit: Human Recognition System with YOLOv8-Pose and OpenCV"
```

## Step 8: Link to GitHub Repository

Replace `YOUR_USERNAME` with your actual GitHub username:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/human-recognition-system.git
```

## Step 9: Push to GitHub

```powershell
git branch -M main
git push -u origin main
```

You'll be prompted to log in to GitHub. Use your:
- **Username**: Your GitHub username
- **Password**: Your GitHub Personal Access Token (not your account password)

### How to Create Personal Access Token:

1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Give it a name: "Git Push Token"
4. Select scopes: Check **"repo"**
5. Click "Generate token"
6. **Copy the token** (you won't see it again!)
7. Use this token as your password when pushing

## Step 10: Verify Upload

1. Go to your GitHub repository: `https://github.com/YOUR_USERNAME/human-recognition-system`
2. You should see all your files!

## Quick Commands Summary

```powershell
# Navigate to project
cd "d:\Mini Project"

# Initialize git
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Human Recognition System"

# Link to GitHub (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/human-recognition-system.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Future Updates

When you make changes to your project:

```powershell
# Add changed files
git add .

# Commit with message
git commit -m "Description of changes"

# Push to GitHub
git push
```

## Troubleshooting

### "fatal: remote origin already exists"
```powershell
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/human-recognition-system.git
```

### Authentication Failed
- Make sure you're using a Personal Access Token, not your password
- Token must have "repo" scope

### Large Files Warning
Some model files might be too large. If you get warnings:
```powershell
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.pt"
git lfs track "*.yml"

# Add and commit
git add .gitattributes
git commit -m "Add Git LFS tracking"
git push
```

---

**You're all set!** Your project is now on GitHub! 🎉
