# Make boxplot of categorical vs. numerical data

# Filter out 'Other' theme
df_filtered <- df[df$primary_theme != "Other", ]

# Calculate median sentiment for each theme to determine sorting order
theme_medians <- aggregate(df$sentiment_compound, by(df$primary_theme),
                           data = df_filtered,
                           FUN = median)
theme_order <- theme_medians$primary_theme[order(theme_medians$sentiment_compound)]

# Create boxplot
boxplot(sentiment_compound ~ factor(primary_theme, levels = theme_order),
        data = df_filtered,
        xlab = "Primary Theme", 
        ylab = "Sentiment Compound Score",
        main = "Sentiment by Theme")
    