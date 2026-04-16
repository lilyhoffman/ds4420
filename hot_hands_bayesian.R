library(tidyverse)
library(brms)
library(bayesplot)
library(tidybayes)

#  Configuration 

DATA_DIR        <- "pbp"
SEASON_START    <- 2004
SEASON_END      <- 2013

PLAYER_ID       <- 977
SHOT_TYPES      <- c("Made Shot", "Missed Shot")
MIN_SHOTS_PER_GAME <- 10

#  Load & concatenate seasons 

load_seasons <- function(start, end) {
  map_dfr(start:end, function(year) {
    path <- file.path(DATA_DIR, paste0("pbp", year, ".csv"))
    if (!file.exists(path)) {
      message("Missing: ", path)
      return(NULL)
    }
    df <- read_csv(path, col_types = cols(gameid = col_character()))
    message(sprintf("Loaded %-30s — %s rows", path, nrow(df)))
    df
  })
}

#  Filter to player shots 

filter_player_shots <- function(df) {
  df |>
    filter(playerid == PLAYER_ID, type %in% SHOT_TYPES) |>
    drop_na(gameid, result, dist) |>
    group_by(gameid) |>
    filter(n() >= MIN_SHOTS_PER_GAME) |>
    ungroup() |>
    mutate(shot_made = as.integer(result == "Made"))
}

#  Feature engineering 

clock_to_seconds <- function(clock_str) {
  map_int(clock_str, function(x) {
    if (is.na(x)) return(NA_integer_)
    m <- str_match(x, "PT(\\d+)M([\\d.]+)S")
    if (is.na(m[1])) return(NA_integer_)
    as.integer(m[2]) * 60L + as.integer(as.double(m[3]))
  })
}

build_streak <- function(made) {
  streak <- integer(length(made))
  count  <- 0L
  for (i in seq_along(made)) {
    streak[i] <- count
    count <- if (made[i] == 1L) count + 1L else 0L
  }
  streak
}

build_miss_streak <- function(made) {
  streak <- integer(length(made))
  count  <- 0L
  for (i in seq_along(made)) {
    streak[i] <- count
    count <- if (made[i] == 0L) count + 1L else 0L
  }
  streak
}

build_shot_features <- function(shots) {
  shots |>
    # Chronological order within game
    arrange(gameid, period, desc(clock)) |>
    group_by(gameid) |>
    mutate(
      # Lag features
      lag_1 = lag(shot_made, 1),
      lag_2 = lag(shot_made, 2),
      lag_3 = lag(shot_made, 3),

      # Rolling hit rates (shift by 1 so current shot not included)
      hit_rate_3  = lag(zoo::rollmeanr(shot_made, k = 3,  fill = NA, align = "right"), 1),
      hit_rate_5  = lag(zoo::rollmeanr(shot_made, k = 5,  fill = NA, align = "right"), 1),
      hit_rate_10 = lag(zoo::rollmeanr(shot_made, k = 10, fill = NA, align = "right"), 1),

      # Streak features
      make_streak = build_streak(shot_made),
      miss_streak = build_miss_streak(shot_made),

      # Shot number within game (0-indexed)
      shot_num = row_number() - 1
    ) |>
    ungroup() |>

    # Drop first shot of each game (no history)
    filter(shot_num > 0) |>

    # Clock in seconds
    mutate(clock_seconds = clock_to_seconds(clock)) |>

    # Bucket rare shot subtypes
    mutate(
      shot_subtype = fct_lump_n(subtype, n = 6, other_level = "Other")
    ) |>

    drop_na(hit_rate_5, make_streak, dist, clock_seconds)
}


#  Run pipeline 




raw     <- load_seasons(SEASON_START, SEASON_END)
shots   <- filter_player_shots(raw)
featured <- build_shot_features(shots)

message("Model-ready rows: ", nrow(featured))



#  Scale continuous features 
# brms can do this inline with scale() — no need for a separate scaler object

model_df <- featured |>
  mutate(
    hit_rate_5_s  = scale(hit_rate_5)[,1],
    make_streak_s = scale(make_streak)[,1],
    dist_s        = scale(dist)[,1],
    clock_s       = scale(clock_seconds)[,1]
  )

#  Priors 

priors <- c(
  prior(student_t(4, 0, 1), class = Intercept),
  prior(normal(0, 1),        class = b)           # applies to all slopes
)

#  Model 

hot_hand_model <- brm(
  formula = shot_made ~ hit_rate_5_s + make_streak_s + dist_s + clock_s + shot_subtype,
  data    = model_df,
  family  = bernoulli(link = "logit"),
  prior   = priors,
  chains  = 4,
  iter    = 2000,       # 1000 warmup + 1000 draws per chain
  warmup  = 1000,
  cores   = 4,
  seed    = 42,
  file    = "hot_hand_model"   # caches compiled model to disk
)

#  Results 

summary(hot_hand_model)

# Trace plots
mcmc_trace(hot_hand_model, pars = c("b_hit_rate_5_s", "b_make_streak_s"))

# Posterior distributions of the hot hand coefficients
hot_hand_model |>
  gather_draws(b_hit_rate_5_s, b_make_streak_s) |>
  ggplot(aes(x = .value, fill = .variable)) +
  geom_density(alpha = 0.6) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  labs(title = "Posterior distributions — hot hand coefficients",
       x = "Coefficient (logit scale)", y = "Density") +
  theme_minimal()

# Probability that each hot hand effect is positive
hot_hand_model |>
  gather_draws(b_hit_rate_5_s, b_make_streak_s) |>
  group_by(.variable) |>
  summarise(p_positive = mean(.value > 0))



  # Posterior probability that each hot hand coefficient is positive
hot_hand_model |>
  gather_draws(b_hit_rate_5_s, b_make_streak_s) |>
  group_by(.variable) |>
  summarise(
    p_positive = mean(.value > 0),
    median     = median(.value),
    ci_lower   = quantile(.value, 0.025),
    ci_upper   = quantile(.value, 0.975)
  )

# Convert a specific scenario to probability
# e.g. what's the predicted P(make) when hit_rate_5 is 1 SD above average?
posterior_epred(hot_hand_model, 
  newdata = data.frame(
    hit_rate_5_s  = c(0, 1),   # average vs. one SD above (i.e. "hot")
    make_streak_s = 0,
    dist_s        = 0,
    clock_s       = 0,
    shot_subtype  = "Jump Shot"
  )
) |> apply(2, mean)



library(broom)


#  1. Fit frequentist GLM to get MLE 
mle_fit <- glm(
  shot_made ~ hit_rate_5_s + make_streak_s + dist_s + clock_s + shot_subtype,
  data   = model_df,
  family = binomial(link = "logit")
)

mle_coefs <- tidy(mle_fit, conf.int = TRUE) |>
  filter(term %in% c("hit_rate_5_s", "make_streak_s")) |>
  mutate(term = recode(term,
    hit_rate_5_s  = "b_hit_rate_5_s",
    make_streak_s = "b_make_streak_s"
  ))

#  Shared white theme override 
white_theme <- theme(
  plot.background  = element_rect(fill = "white", color = NA),
  panel.background = element_rect(fill = "white", color = NA),
  legend.background = element_rect(fill = "white", color = NA)
)

#  1. Trace plots 
trace_p <- mcmc_trace(hot_hand_model, pars = c("b_hit_rate_5_s", "b_make_streak_s")) +
  white_theme +
  labs(title = "Trace plots — hot hand coefficients")

ggsave("plot_trace.png", plot = trace_p, width = 10, height = 4, dpi = 150, bg = "white")

#  2. Posterior density 
posterior_p <- hot_hand_model |>
  gather_draws(b_hit_rate_5_s, b_make_streak_s) |>
  ggplot(aes(x = .value, fill = .variable)) +
  geom_density(alpha = 0.6) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  labs(title = "Posterior distributions — hot hand coefficients",
       x = "Coefficient (logit scale)", y = "Density") +
  theme_minimal() +
  white_theme

ggsave("plot_posterior.png", plot = posterior_p, width = 8, height = 4, dpi = 150, bg = "white")

#  3. Prior vs MLE vs Posterior 

posterior_df <- hot_hand_model |>
  gather_draws(b_hit_rate_5_s, b_make_streak_s)

# Zoom x-axis to where the posterior actually lives
x_range <- range(posterior_df$.value) * 1.3

prior_df_zoom <- tibble(
  x = seq(x_range[1], x_range[2], length.out = 500),
  y = dnorm(x, mean = 0, sd = 1)   # prior is still Normal(0,1), just plotted on zoomed range
)
# Zoom x-axis to where the posterior actually lives
x_range <- range(posterior_df$.value) * 1.3

prior_df_zoom <- tibble(
  x = seq(x_range[1], x_range[2], length.out = 500),
  y = dnorm(x, mean = 0, sd = 1)   # prior is still Normal(0,1), just plotted on zoomed range
)

prior_mle_p <- ggplot() +
  geom_line(
    data = prior_df_zoom,
    aes(x = x, y = y, linetype = "Prior  Normal(0,1)"),
    color = "grey40", linewidth = 0.8
  ) +
  geom_density(
    data = posterior_df,
    aes(x = .value, fill = "Posterior"),
    alpha = 0.35, color = NA
  ) +
  geom_vline(
    data = mle_coefs,
    aes(xintercept = estimate, color = "MLE"),
    linewidth = 0.9
  ) +
  geom_rect(
    data = mle_coefs,
    aes(xmin = conf.low, xmax = conf.high,
        ymin = -Inf, ymax = Inf, fill = "MLE 95% CI"),
    alpha = 0.15, inherit.aes = FALSE
  ) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black", linewidth = 0.5) +
  facet_wrap(~ .variable, labeller = as_labeller(c(
    b_hit_rate_5_s  = "Rolling hit rate (5-shot window)",
    b_make_streak_s = "Make streak"
  ))) +
  scale_color_manual(name = NULL, values = c("MLE" = "#E63946")) +
  scale_fill_manual(name = NULL, values = c(
    "Posterior"  = "#457B9D",
    "MLE 95% CI" = "#E63946"
  )) +
  scale_linetype_manual(name = NULL, values = c("Prior  Normal(0,1)" = "solid")) +
  coord_cartesian(xlim = x_range) +   # zoom without dropping data
  labs(
    title    = "Prior vs. MLE — hot hand coefficients",
    subtitle = "Prior is Normal(0,1) — shown at posterior scale; it is essentially flat here",
    x        = "Coefficient (logit scale)",
    y        = "Density"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom") +
  white_theme

ggsave("plot_prior_mle.png", plot = prior_mle_p, width = 10, height = 5, dpi = 150, bg = "white")

message("All plots saved to: ", getwd())