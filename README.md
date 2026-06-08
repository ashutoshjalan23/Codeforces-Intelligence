# Codeforces Intelligence Dashboard

A Jupyter notebook for analyzing and visualizing your Codeforces competitive programming performance.

## Overview

This project provides comprehensive analysis of:
- **Rating Progression**: Track how your official rating changes over contest history
- **Practice Patterns**: Analyze problem difficulty distribution during practice
- **Skill Metrics**: Evaluate performance gaps and identify strong/weak topics
- **Topic Analysis**: See which algorithmic topics you excel at

## Quick Start

### Requirements
- Python 3.8+
- Jupyter Notebook/Lab
- Libraries: `requests`, `pandas`, `matplotlib`, `seaborn`, `numpy`

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Start Jupyter
jupyter notebook
```

### Usage

1. Open `cpintelligence_clean.ipynb`
2. Update the `USERNAME` variable in the first code cell with your Codeforces handle
3. Run all cells sequentially

## Features

### Data Source
- Fetches real-time data from the official [Codeforces API](https://codeforces.com/apiHelp)
- No authentication required for public profiles

### Visualizations

**Panel 1: Rating Progression**
- Line chart showing official rating changes
- Visual markers for rating tier boundaries (Pupil: 1200, Specialist: 1400)

**Panel 2: Practice Difficulty**
- Scatter plot of individual solved problems
- 20-problem moving average trend line
- Shows difficulty growth relative to contest ratings

**Panel 3: Performance Gap**
- Measures (Problem Rating - User Rating)
- Positive gap = practicing above current level
- Useful indicator of training intensity

**Panel 4: Topic Skills**
- Bar chart of top 10 problem topics
- Ranked by average solved difficulty
- Identifies strengths and improvement areas

## Example Output

The notebook generates a 4-panel dashboard showing:
- Time-based rating growth curve
- Practice volume and intensity trends
- Performance deficit/surplus analysis
- Topic-based skill ranking

## Data Processing

1. **Deduplication**: Removes duplicate submissions of same problem
2. **Filtering**: Keeps only successful submissions (verdict = "OK")
3. **Temporal Alignment**: Matches practice difficulties to contest ratings
4. **Aggregation**: Calculates statistics by topic and time period

## Metrics Explained

- **Average Problem Rating**: Mean difficulty of solved problems
- **Performance Gap**: How far ahead/behind you practice relative to current rating
- **Rolling Average**: 20-problem moving average of practice difficulty
- **Topic Skills**: Average difficulty solved per problem category

## Files

- `cpintelligence_clean.ipynb` - Main analysis notebook (ready for GitHub)
- `README.md` - This file
- `requirements.txt` - Python dependencies

## Notes

- Data is fetched real-time from Codeforces API
- Deduplication ensures each problem is counted once
- All timestamps are converted to local timezone
- Missing ratings are handled gracefully

## Customization

Edit the notebook to:
- Change plot colors and styles
- Add additional statistical measures
- Filter by date range
- Export visualizations as images

## License

MIT License - Feel free to use and modify

## Contributing

Pull requests and suggestions welcome!

## Disclaimer

This tool is for personal analysis only. Always respect Codeforces API rate limits (not more than 5 requests per second).
