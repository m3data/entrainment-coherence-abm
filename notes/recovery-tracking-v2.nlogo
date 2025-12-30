;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; COHERENCE vs ENTRAINMENT MODEL
;; Illustrative agent-based dynamical sketch
;; Recovery Tracking v2
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

globals [
  perturbing?              ;; whether a perturbation is active
  perturb-timer
  max-variance
  perturb-started?
  last-perturb-strength
  perturb-schedule

  ;; Recovery tracking v2
  stable-baseline          ;; rolling average before perturbation
  baseline-window          ;; list for rolling calculation
  perturb-end-tick         ;; when perturbation finished
  peak-deviation           ;; max deviation from baseline during/after perturbation
  recovery-complete?       ;; flag for clean recovery detection
  last-recovery-time       ;; ticks from perturbation end to tolerance return
]

turtles-own [
  preferred-heading    ;; internal tendency (identity proxy)
  coupling-bias        ;; individual sensitivity to alignment (0-1)
  inertia              ;; heading update inertia (0–1)
  tie-strength         ;; how influential this turtle is to others (0–1)
]

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; SETUP
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

to setup
  clear-all
  set max-variance 180
  set perturbing? false
  set perturb-timer 0
  set perturb-started? false
  set last-perturb-strength 0

  ;; Recovery tracking v2 init
  set baseline-window []
  set stable-baseline 0
  set perturb-end-tick 0
  set peak-deviation 0
  set recovery-complete? true
  set last-recovery-time -1  ;; -1 means no recovery measured yet

  create-turtles population [
    setxy random-xcor random-ycor
    set heading random 360
    set preferred-heading heading
    set coupling-bias random-float 1
    set inertia (0.2 + random-float 0.8)
    set tie-strength (0.2 + random-float 0.8)
    set shape "agent-field"
    set color scale-color blue coupling-bias 0 1
    set size 1 + (2 * tie-strength)
  ]

  ;; Perturbation schedule
  set perturb-schedule []

  if perturbation-regime = "single" [
    set perturb-schedule (list 300)
  ]

  if perturbation-regime = "periodic" [
    set perturb-schedule n-values 10 [ i -> 300 + (i * 300) ]
  ]

  if perturbation-regime = "irregular" [
    set perturb-schedule sort n-of 5 (range 200 2000)
  ]

  ;; --- plot initialisation ---
  set-current-plot "Heading Variance Over Time"
  clear-plot

  reset-ticks
end

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; MAIN LOOP
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

to go
  ;; --- Update rolling baseline (only when stable: not perturbing and recovered) ---
  if (not perturbing?) and recovery-complete? [
    set baseline-window lput heading-variance baseline-window
    ;; Keep window at 50 ticks max
    if length baseline-window > 50 [
      set baseline-window but-first baseline-window
    ]
    ;; Only compute baseline once we have enough samples
    if length baseline-window >= 10 [
      set stable-baseline mean baseline-window
    ]
  ]

  ;; --- Apply ongoing perturbation ---
  if perturbing? [
    apply-perturbation
  ]

  ;; --- Scheduled perturbations ---
  if member? ticks perturb-schedule [
    perturb
  ]

  ;; --- Agent dynamics ---
  ask turtles [
    update-heading
    apply-noise
    move
  ]

  ;; --- Track peak deviation during and after perturbation ---
  if (not recovery-complete?) [
    let current-deviation abs (heading-variance - stable-baseline)
    if current-deviation > peak-deviation [
      set peak-deviation current-deviation
    ]
  ]

  ;; --- Recovery tracking v2 ---
  ;; Recovery is measured from perturbation END to when variance returns within tolerance of stable baseline
  if (not recovery-complete?) and (not perturbing?) [
    let current-deviation abs (heading-variance - stable-baseline)

    ;; Recovery complete when within tolerance band of stable baseline
    if current-deviation <= recovery-tolerance [
      set last-recovery-time (ticks - perturb-end-tick)
      set recovery-complete? true
    ]
  ]

  ;; --- Plotting ---
  set-current-plot "Heading Variance Over Time"
  set-current-plot-pen "variance"
  plot heading-variance

  set-current-plot-pen "perturb"
  if perturb-started? [
    plot-pen-reset
    plotxy ticks 0
    plotxy ticks ((last-perturb-strength / 360) * max-variance)
    set perturb-started? false
  ]

  tick
end

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; DYNAMICS
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

to update-heading
  let nearby-turtles turtles in-radius 3

  if any? nearby-turtles [
    let total-weight sum [tie-strength] of nearby-turtles

    let weighted-x sum [ tie-strength * sin heading ] of nearby-turtles
    let weighted-y sum [ tie-strength * cos heading ] of nearby-turtles
    let avg-heading atan weighted-x weighted-y

    let desired-turn 0

    if entrainment-mode? [
      set desired-turn
        (coupling-strength * coupling-bias *
         subtract-headings avg-heading heading)
    ]

    if not entrainment-mode? [
      let identity-pull subtract-headings preferred-heading heading
      let social-pull subtract-headings avg-heading heading

      set desired-turn
        (coupling-strength * coupling-bias * 0.5 * social-pull)
        + (0.2 * identity-pull)
    ]

    set heading heading + ((1 - inertia) * desired-turn)
  ]
end

to apply-noise
  set heading heading + (noise-level * (random-float 2 - 1) * 30)
end

to move
  fd 0.3
end

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; PERTURBATION
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

to perturb
  ;; Only start new perturbation if not already perturbing
  if not perturbing? [
    set perturbing? true
    set perturb-timer perturb-duration
    set perturb-started? true
    set last-perturb-strength perturbation-strength

    ;; Reset recovery tracking for this perturbation
    set peak-deviation 0
    set recovery-complete? false
    ;; NOTE: stable-baseline is NOT reset - it retains the rolling pre-perturbation value
  ]
end

to apply-perturbation
  ask turtles [
    set heading heading + (random-float perturbation-strength - perturbation-strength / 2)
  ]

  set perturb-timer perturb-timer - 1
  if perturb-timer <= 0 [
    set perturbing? false
    set perturb-end-tick ticks  ;; Mark when perturbation ended - recovery timing starts here
  ]
end

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; REPORTERS
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

to-report heading-variance
  if not any? turtles [
    report 0
  ]

  let mean-heading mean [heading] of turtles
  report mean [ abs subtract-headings heading mean-heading ] of turtles
end

;; Additional reporters for BehaviorSpace experiments
to-report recovery-time
  ;; Returns last-recovery-time, or -1 if recovery not yet complete
  report last-recovery-time
end

to-report baseline
  ;; Returns the stable baseline variance (rolling average)
  report stable-baseline
end

to-report max-deviation
  ;; Returns peak deviation from baseline during/after perturbation
  report peak-deviation
end

to-report is-recovering?
  ;; Returns true if system is still recovering from perturbation
  report not recovery-complete?
end
