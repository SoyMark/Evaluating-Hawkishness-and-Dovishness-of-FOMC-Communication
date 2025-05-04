import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


def regression_for_similarity_vector(corpus_type):
    # Load data
    X_df = pd.read_csv(f"result/{corpus_type}/similarity_vector.csv", parse_dates=["date"])
    y_df = pd.read_csv("data/treasury_spread.csv", parse_dates=["date"])

    # Merge and preprocess
    df = pd.merge(X_df, y_df, on="date", how="inner").dropna()
    df["year"] = df["date"].dt.year
    df["hawkish_minus_dovish"] = df["avg_cosine_with_hawkish"] - df["avg_cosine_with_dovish"]
    # print(df.head())
    # Prepare plot
    plt.figure(figsize=(9, 7))
    palette = {2019: "blue", 2024: "yellow"}

    for selected_year in [2019, 2024]:
        if selected_year == 2019:
            df_sub = df[df["year"] <= 2019]
        else:
            df_sub = df[df["year"] >= 2024]

        if df_sub.empty:
            print(f"No data for year {selected_year}. Skipping regression.")
            continue  # Skip if no data

        X = df_sub[["avg_cosine_with_dovish", "avg_cosine_with_hawkish"]]
        y = df_sub["90-day Treasury/5-year yield spread"]

        # Linear regression
        model = LinearRegression()
        model.fit(X, y)
        coef = model.coef_
        intercept = model.intercept_
        y_pred = model.predict(X)

        print(f"\n==== Results for Year {'2018-2019' if selected_year==2019 else '2024-2025'} ====")
        print(f"  dovish:  {coef[0]:.6f}")
        print(f"  hawkish: {coef[1]:.6f}")
        print(f"Intercept: {intercept:.6f}")
        print("R^2 Score:", r2_score(y, y_pred))
        print("MSE:", mean_squared_error(y, y_pred))

        # Scatter plot with regression line
        sns.regplot(
            x=df_sub["hawkish_minus_dovish"], 
            y=y,
            ci=None,
            label=f"{'2018-2019' if selected_year==2019 else '2024-2025'}",
            scatter_kws={"s": 50, "alpha": 0.6},
            line_kws={"linewidth": 2},
            color=palette[selected_year]
        )

    # Plot aesthetics
    plt.xlabel("Hawkish over Dovish (similarity method)")
    plt.ylabel("90-day Treasury/5-year Yield Spread")
    plt.title(f"Spread vs. Hawkish-Dovish Shift in {corpus_type} Data")
    plt.legend(title="Year Range")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def regression_for_classification_vector(corpus_type):
    # Load and parse dates
    X_df = pd.read_csv(f"result/{corpus_type}/classification_vector.csv", parse_dates=["date"])
    Y_df = pd.read_csv("data/treasury_spread.csv", parse_dates=["date"])

    # Merge (backward join) to align classification vector with closest past spread
    merged_df = pd.merge_asof(
        Y_df.sort_values("date"),
        X_df.sort_values("date"),
        on="date",
        direction="backward"
    )
    merged_df = merged_df.dropna(subset=["dovish", "hawkish", "90-day Treasury/5-year yield spread"])
    merged_df["year"] = merged_df["date"].dt.year
    merged_df["hawkish_minus_dovish"] = merged_df["hawkish"] - merged_df["dovish"]

    # Prepare plot
    plt.figure(figsize=(9, 7))
    palette = {2019: "blue", 2024: "orange"}

    for selected_year in [2019, 2024]:
        if selected_year == 2019:
            df_sub = merged_df[merged_df["year"] <= 2019]
        else:
            df_sub = merged_df[merged_df["year"] >= 2024]

        if df_sub.empty:
            print(f"No data for year {selected_year}. Skipping regression.")
            continue

        X = df_sub[["dovish", "hawkish"]]
        y = df_sub["90-day Treasury/5-year yield spread"]

        # Linear regression
        model = LinearRegression()
        model.fit(X, y)
        coef = model.coef_
        intercept = model.intercept_
        y_pred = model.predict(X)

        print(f"\n==== Results for Year {'2018-2019' if selected_year==2019 else '2024-2025'} ====")
        print(f"  dovish:  {coef[0]:.6f}")
        print(f"  hawkish: {coef[1]:.6f}")
        print(f"Intercept: {intercept:.6f}")
        print("R^2 Score:", r2_score(y, y_pred))
        print("MSE:", mean_squared_error(y, y_pred))

        # Plot with regplot
        sns.regplot(
            x=df_sub["hawkish_minus_dovish"],
            y=y,
            ci=None,
            label=f"{'2018-2019' if selected_year==2019 else '2024-2025'}",
            scatter_kws={"s": 50, "alpha": 0.6},
            line_kws={"linewidth": 2},
            color=palette[selected_year]
        )

    # Plot aesthetics
    plt.xlabel("Hawkish over Dovish (classification method)")
    plt.ylabel("90-day Treasury/5-year Yield Spread")
    plt.title(f"Spread vs. Hawkish-Dovish Shift in {corpus_type} Data")
    plt.legend(title="Year Range")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    for type in ["fomc_minutes", "press_conferences", "fed_speeches"]:
        regression_for_classification_vector(type)
        regression_for_similarity_vector(type)