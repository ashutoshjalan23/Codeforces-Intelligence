# Quick Start Guide

## 1. Setup Your Environment

```bash
# Navigate to project directory
cd cptracker

# Create virtual environment (optional but recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Configure Your Handle

Open `cpintelligence_clean.ipynb` and change:

```python
USERNAME = "your_codeforces_handle"
```

Replace with your actual Codeforces username (e.g., "jalanashutosh06").

## 3. Run the Notebook

```bash
# Start Jupyter
jupyter notebook

# Then open cpintelligence_clean.ipynb in your browser
```

Or use Jupyter Lab:

```bash
jupyter lab
```

## 4. Execute Cells

- Click on a cell and press `Shift+Enter` to run it
- Run all cells from top to bottom
- Wait for API calls to complete (usually 5-10 seconds)

## 5. View Results

Once complete, you'll see:
- ✓ Status messages showing data fetched
- ✓ Statistics table with topic breakdown
- ✓ 4-panel dashboard with visualizations

## Troubleshooting

**Issue: "API Error: handle ..."**
- Check your username is spelled correctly (case-sensitive)
- Verify the profile is public on Codeforces

**Issue: "Network error"**
- Check your internet connection
- Try again after a few seconds (API rate limit?)

**Issue: Empty charts**
- Ensure your profile has problems solved
- Check username is correct

## Tips

- The notebook works entirely offline after first run
- Save outputs: `File → Download as → PDF`
- Modify chart colors in matplotlib code
- Add filters for date ranges or problem difficulty

## Next Steps

- Push to GitHub: `git add . && git commit -m "Initial commit" && git push`
- Share your analysis
- Track progress over time by re-running the notebook
