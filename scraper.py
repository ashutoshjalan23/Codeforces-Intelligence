

from intelligence import analyze_user, extract_ml_features
import json
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment

def analyze_batch(usernames, output_file="analysis_results.json"):
    """

    """
    results = {}
    
    for i, username in enumerate(usernames, 1):
        print(f"\n[{i}/{len(usernames)}] Analyzing {username}...")
        
        result = analyze_user(username)
        
        if result["success"]:
            stats = result["stats"]
            # Convert to JSON-serializable format
            results[username] = {
                "current_rating": stats.get("current_rating"),
                "peak_rating": stats.get("peak_rating"),
                "total_problems": stats.get("total_problems"),
                "avg_problem_rating": stats.get("avg_problem_rating"),
                "contest_count": stats.get("contest_count"),
                "avg_gap": stats.get("avg_gap")
            }
            
            print(f"✓ Rating: {stats.get('current_rating')}")
            print(f"✓ Problems Solved: {stats.get('total_problems')}")
        else:
            print(f"✗ Failed: {result.get('error')}")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    return results


def extract_ml_dataset(usernames, output_file="ml_dataset.json"):

    ml_data = []
    
    for i, username in enumerate(usernames, 1):
        print(f"[{i}/{len(usernames)}] Extracting features for {username}...")
        
        result = analyze_user(username)
        
        if result["success"]:
            features = extract_ml_features(
                username,
                result["stats"],
                result["df_practice"],
                result["df_contests"]
            )
            ml_data.append(features)
            print(f"Rating: {features['current_rating']}, Problems: {features['total_problems']}")
        else:
            print(f"Failed: {result.get('error')}")
    
    with open(output_file, "w") as f:
        json.dump(ml_data, f, indent=2)
    
    print(f"\n ML features saved to {output_file}")
    return ml_data


def ml_data_to_excel_long_format(ml_data, output_file="user_topic_features.xlsx"):

    rows = []
    
    for user in ml_data:
        base_features = {
            "username": user["username"],
            "current_rating": user["current_rating"],
            "peak_rating": user["peak_rating"],
            "gap_to_peak": user["gap_to_peak"],
            "total_problems": user["total_problems"],
            "avg_problem_rating": user["avg_problem_rating"],
            "avg_gap": user["avg_gap"],
            "contest_count": user["contest_count"],
            "avg_rating_change": user["avg_rating_change"],
            "rating_volatility": user["rating_volatility"],
            "rating_trend": user["rating_trend"],
            "contest_problems_count": user["contest_problems_count"],
            "practice_problems_count": user["practice_problems_count"],
            "avg_practice_problems_before_contests": user["avg_practice_problems_before_contests"],
            "avg_rating_diff_practice_vs_contest": user["avg_rating_diff_practice_vs_contest"],
        }
        
        for topic_name, topic_data in user["topic_stats"].items():
            row = base_features.copy()
            row.update({
                "topic": topic_name,
                "topic_problems_solved": topic_data["problems_solved"],
                "topic_avg_rating": topic_data["avg_rating"],
                "topic_max_rating": topic_data["max_rating"],
                "topic_avg_gap": topic_data["avg_gap"],
                "topic_difficulty_gap": topic_data["difficulty_gap"],
            })
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Write to Excel with formatting
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Training Data', index=False)
        
        worksheet = writer.sheets['Training Data']
        
        # Format header
        header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        header_font = Font(bold=True, color="FFFFFF")
        
        for cell in worksheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")
        
        # Adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    print(f"✓ Excel (long format) saved to {output_file}")
    print(f"  Total rows: {len(df)}")
    print(f"  Format: Each row = 1 user + 1 topic combination")
    return df




if __name__ == "__main__":
  
    print("\n" + "=" * 50)
    print(" Batch Scraping from usernames.txt")
    print("=" * 50)
    
    try:
        with open("usernames.txt", "r") as f:
            usernames = [line.strip() for line in f if line.strip()]
        
        
    
        analyze_batch(usernames)
        
    
        print("\n" + "=" * 50)
        print("Example 3: Extract ML Features for Training")
        print("=" * 50)
        
        ml_data = extract_ml_dataset(usernames)
        
        # Convert to Excel - Long format (RECOMMENDED for XGBoost)
        print("\n[Format 1] Long format (1 row per user-topic):")
        df_long = ml_data_to_excel_long_format(ml_data, "training_data_long.xlsx")
        print(f"Shape: {df_long.shape}")
        print(f"Columns: {list(df_long.columns)}\n")
        

 
        
    except FileNotFoundError:
        print("usernames.txt not found. Please run scraper.py first.")
