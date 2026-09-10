#
# Check A extraction (does not modify authors' scripts or data).
# Re-estimates the headline ITT specification from main_analysis.R
# and writes checkA tables + coefficient plot.
#

suppressPackageStartupMessages({
  library(readr)
  library(sandwich)
})

data_path <- "authors/final_data.csv"
out_dir <- "."

df_raw <- read_csv(data_path, show_col_types = FALSE)
n_raw <- nrow(df_raw)
n_honors <- sum(df_raw$Honors == 1, na.rm = TRUE)
n_nonhonors <- sum(df_raw$Honors == 0, na.rm = TRUE)

df <- df_raw[df_raw$Honors == 0, ]
df$teacher <- as.factor(df$teacher)
df$Session <- as.factor(df$Session)
df$Year <- as.factor(df$Year)
df$Grader <- as.factor(df$Grader)

# Same specification as authors' main ITT (Eq. 1 / Table 1)
reg2 <- lm(Part2Tot ~ GPTBase + GPTTutor + gpa_prev +
             teacher + Session + Grader + Year, data = df)
reg3 <- lm(Part3Tot ~ GPTBase + GPTTutor + gpa_prev +
             teacher + Session + Grader + Year, data = df)

vc2 <- vcovCL(reg2, cluster = ~Class)
vc3 <- vcovCL(reg3, cluster = ~Class)

extract_term <- function(model, vcov_mat, term, df_t = NULL) {
  b <- unname(coef(model)[term])
  se <- sqrt(diag(vcov_mat))[term]
  if (is.null(df_t)) df_t <- df.residual(model)
  tstat <- b / se
  p <- 2 * pt(-abs(tstat), df = df_t)
  tcrit <- qt(0.975, df = df_t)
  list(
    estimate = b,
    se = unname(se),
    t = unname(tstat),
    p_value = unname(p),
    ci_low = b - tcrit * se,
    ci_high = b + tcrit * se,
    n = nobs(model),
    df_t = df_t
  )
}

extract_contrast <- function(model, vcov_mat, term_a, term_b, df_t = NULL) {
  # term_a - term_b
  ba <- unname(coef(model)[term_a])
  bb <- unname(coef(model)[term_b])
  b <- ba - bb
  va <- vcov_mat[term_a, term_a]
  vb <- vcov_mat[term_b, term_b]
  cab <- vcov_mat[term_a, term_b]
  se <- sqrt(va + vb - 2 * cab)
  if (is.null(df_t)) df_t <- df.residual(model)
  tstat <- b / se
  p <- 2 * pt(-abs(tstat), df = df_t)
  tcrit <- qt(0.975, df = df_t)
  list(
    estimate = b,
    se = unname(se),
    t = unname(tstat),
    p_value = unname(p),
    ci_low = b - tcrit * se,
    ci_high = b + tcrit * se,
    n = nobs(model),
    df_t = df_t
  )
}

fmt <- function(x, d = 6) formatC(x, digits = d, format = "f")

p2_base <- extract_term(reg2, vc2, "GPTBase")
p2_tutor <- extract_term(reg2, vc2, "GPTTutor")
p3_base <- extract_term(reg3, vc3, "GPTBase")
p3_tutor <- extract_term(reg3, vc3, "GPTTutor")
p3_tutor_vs_base <- extract_contrast(reg3, vc3, "GPTTutor", "GPTBase")

used <- complete.cases(df[, c("Part2Tot", "Part3Tot", "GPTBase", "GPTTutor",
                              "gpa_prev", "teacher", "Session", "Grader", "Year", "Class")])
n_clusters <- length(unique(df$Class[used]))
n_dropped_missing <- n_nonhonors - sum(used)

spec_text <- paste(
  "OLS ITT; Honors==0; covariates: gpa_prev;",
  "FE: teacher, Session, Grader, Year;",
  "cluster-robust SE (sandwich::vcovCL) at Class"
)

published <- list(
  p2_base = list(est = 0.137, se = 0.031),
  p2_tutor = list(est = 0.361, se = 0.032),
  p3_base = list(est = -0.054, se = 0.022),
  p3_tutor = list(est = -0.004, se = 0.013)
)

matches_published <- function(got, pub, tol = 0.0005) {
  # stargazer Table 1 is rounded to 3 decimals
  (abs(round(got$estimate, 3) - pub$est) < tol) &&
    (abs(round(got$se, 3) - pub$se) < tol)
}

match_note <- function(got, pub) {
  if (matches_published(got, pub)) {
    "yes: matches PNAS Table 1 when rounded to 3 decimals"
  } else {
    sprintf(
      "discrepancy: reproduced %.6f (se %.6f) vs published %.3f (se %.3f)",
      got$estimate, got$se, pub$est, pub$se
    )
  }
}

fe_cov <- "FE: teacher, Session, Grader, Year; covariate: gpa_prev"
cluster <- "Class (sandwich::vcovCL default; 44 classrooms in analysis sample)"

log_rows <- data.frame(
  result_id = c(
    "A_assisted_GPTBase_vs_Control",
    "A_assisted_GPTTutor_vs_Control",
    "B_unassisted_GPTBase_vs_Control",
    "B_unassisted_GPTTutor_vs_Control"
  ),
  headline = c(
    "A. performance while AI assistance is available",
    "A. performance while AI assistance is available",
    "B. later performance when AI assistance is removed",
    "B. later performance when AI assistance is removed"
  ),
  original_script = "main_regressions/main_analysis.R",
  dataset = "main_regressions/final_data.csv",
  dependent_variable = c("Part2Tot", "Part2Tot", "Part3Tot", "Part3Tot"),
  treatment_comparison = c(
    "GPT Base (vanilla) vs Control",
    "GPT Tutor (augmented) vs Control",
    "GPT Base (vanilla) vs Control",
    "GPT Tutor (augmented) vs Control"
  ),
  coefficient = c(p2_base$estimate, p2_tutor$estimate, p3_base$estimate, p3_tutor$estimate),
  standard_error = c(p2_base$se, p2_tutor$se, p3_base$se, p3_tutor$se),
  ci_low = c(p2_base$ci_low, p2_tutor$ci_low, p3_base$ci_low, p3_tutor$ci_low),
  ci_high = c(p2_base$ci_high, p2_tutor$ci_high, p3_base$ci_high, p3_tutor$ci_high),
  p_value = c(p2_base$p_value, p2_tutor$p_value, p3_base$p_value, p3_tutor$p_value),
  n = c(p2_base$n, p2_tutor$n, p3_base$n, p3_tutor$n),
  n_clusters = n_clusters,
  fixed_effects_covariates = fe_cov,
  clustering = cluster,
  published_coefficient = c(0.137, 0.361, -0.054, -0.004),
  published_se = c(0.031, 0.032, 0.022, 0.013),
  matches_published = c(
    match_note(p2_base, published$p2_base),
    match_note(p2_tutor, published$p2_tutor),
    match_note(p3_base, published$p3_base),
    match_note(p3_tutor, published$p3_tutor)
  ),
  stringsAsFactors = FALSE
)

write.csv(log_rows, file.path(out_dir, "reproduction_log.csv"), row.names = FALSE)

results <- data.frame(
  outcome = c(
    "assisted_practice_Part2Tot",
    "assisted_practice_Part2Tot",
    "later_unassisted_exam_Part3Tot",
    "later_unassisted_exam_Part3Tot",
    "later_unassisted_exam_Part3Tot"
  ),
  contrast = c(
    "GPT Base vs Control",
    "GPT Tutor vs Control",
    "GPT Base vs Control",
    "GPT Tutor vs Control",
    "GPT Tutor vs GPT Base"
  ),
  estimate = c(
    p2_base$estimate, p2_tutor$estimate,
    p3_base$estimate, p3_tutor$estimate, p3_tutor_vs_base$estimate
  ),
  se = c(
    p2_base$se, p2_tutor$se,
    p3_base$se, p3_tutor$se, p3_tutor_vs_base$se
  ),
  ci_low = c(
    p2_base$ci_low, p2_tutor$ci_low,
    p3_base$ci_low, p3_tutor$ci_low, p3_tutor_vs_base$ci_low
  ),
  ci_high = c(
    p2_base$ci_high, p2_tutor$ci_high,
    p3_base$ci_high, p3_tutor$ci_high, p3_tutor_vs_base$ci_high
  ),
  p_value = c(
    p2_base$p_value, p2_tutor$p_value,
    p3_base$p_value, p3_tutor$p_value, p3_tutor_vs_base$p_value
  ),
  n = c(
    p2_base$n, p2_tutor$n,
    p3_base$n, p3_tutor$n, p3_tutor_vs_base$n
  ),
  specification = spec_text,
  stringsAsFactors = FALSE
)

write.csv(results, file.path(out_dir, "checkA_results.csv"), row.names = FALSE)

# Coefficient plot
png(file.path(out_dir, "checkA_coefficient_plot.png"),
    width = 2400, height = 1400, res = 200)

par(mfrow = c(1, 2), mar = c(5, 11, 3.5, 1.5), oma = c(2.2, 0, 2.2, 0))

plot_panel <- function(ests, ses, lows, highs, labels, title) {
  n <- length(ests)
  y <- n:1
  xlim <- range(c(lows, highs, 0))
  xlim <- xlim + c(-0.03, 0.03) * diff(xlim)
  plot(NA, xlim = xlim, ylim = c(0.5, n + 0.5),
       xlab = "Estimate (score points, 0-1 scale)", ylab = "",
       yaxt = "n", main = title, cex.main = 1.05)
  abline(v = 0, lty = 2, col = "gray40")
  arrows(lows, y, highs, y, code = 3, angle = 90, length = 0.06, lwd = 2)
  points(ests, y, pch = 16, cex = 1.4)
  axis(2, at = y, labels = labels, las = 1, cex.axis = 0.9)
  text(ests, y + 0.28,
       sprintf("%.3f", ests), cex = 0.75, col = "gray20")
}

plot_panel(
  ests = c(p2_tutor$estimate, p2_base$estimate),
  ses = c(p2_tutor$se, p2_base$se),
  lows = c(p2_tutor$ci_low, p2_base$ci_low),
  highs = c(p2_tutor$ci_high, p2_base$ci_high),
  labels = c("GPT Tutor vs Control", "GPT Base vs Control"),
  title = "Assisted / current performance\n(Part 2 practice)"
)

plot_panel(
  ests = c(p3_tutor_vs_base$estimate, p3_tutor$estimate, p3_base$estimate),
  ses = c(p3_tutor_vs_base$se, p3_tutor$se, p3_base$se),
  lows = c(p3_tutor_vs_base$ci_low, p3_tutor$ci_low, p3_base$ci_low),
  highs = c(p3_tutor_vs_base$ci_high, p3_tutor$ci_high, p3_base$ci_high),
  labels = c("GPT Tutor vs GPT Base", "GPT Tutor vs Control", "GPT Base vs Control"),
  title = "Later unassisted performance\n(Part 3 exam)"
)

mtext("Check A: Bastani et al. ITT contrasts (authors' original specification)",
      outer = TRUE, cex = 1.05, font = 2)
mtext("95% CIs from cluster-robust SEs at Class; t critical value with residual df. Outcomes not differenced across assessments.",
      side = 1, outer = TRUE, cex = 0.75)

dev.off()

# Session diagnostics for notes
ctrl <- df$`Treatment arm` == "control"
van <- df$`Treatment arm` == "vanilla"
aug <- df$`Treatment arm` == "augmented"

diag <- list(
  n_raw = n_raw,
  n_honors = n_honors,
  n_nonhonors = n_nonhonors,
  n_regression = nobs(reg2),
  n_dropped_missing = n_dropped_missing,
  n_clusters = n_clusters,
  n_control = sum(ctrl, na.rm = TRUE),
  n_vanilla = sum(van, na.rm = TRUE),
  n_augmented = sum(aug, na.rm = TRUE),
  mean_p2_control = mean(df$Part2Tot[ctrl], na.rm = TRUE),
  mean_p2_vanilla = mean(df$Part2Tot[van], na.rm = TRUE),
  mean_p2_augmented = mean(df$Part2Tot[aug], na.rm = TRUE),
  mean_p3_control = mean(df$Part3Tot[ctrl], na.rm = TRUE),
  mean_p3_vanilla = mean(df$Part3Tot[van], na.rm = TRUE),
  mean_p3_augmented = mean(df$Part3Tot[aug], na.rm = TRUE),
  p2_base = p2_base,
  p2_tutor = p2_tutor,
  p3_base = p3_base,
  p3_tutor = p3_tutor,
  p3_tutor_vs_base = p3_tutor_vs_base
)

saveRDS(diag, file.path(out_dir, "checkA_diagnostics.rds"))
cat("Wrote checkA tables and plot.\n")
print(results)
cat("\nTutor vs Base (Part 3):\n")
print(p3_tutor_vs_base)
cat("\nN raw / honors / nonhonors / regression / clusters:\n")
cat(n_raw, n_honors, n_nonhonors, nobs(reg2), n_clusters, "\n")

