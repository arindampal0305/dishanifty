# APPLICATION_GUIDE.md — Disha, explained in plain words

This guide describes what Disha is and what every part of the screen does. It
assumes no knowledge of programming, statistics or the stock market. If you want
the technical walkthrough instead, read `DATA_FLOW.md`.

---

## What Disha is

Disha is a desktop app that looks at the past ten years of an Indian stock's
price history and makes one guess about tomorrow:

> **Will it close higher, lower, or roughly the same?**

That is the whole question. Three possible answers — **UP**, **DOWN**, or
**NEUTRAL** — and nothing else. Disha never guesses a price, never says how much
something will move, and never says anything about next week or next year.

It was built as a college project to practise doing this kind of prediction
*correctly*. One of its most useful outputs is the honest answer to "does this
actually work?" — and that answer, stated openly inside the app, is **mostly
no**. More on that at the end, because it is the most important section here.

**Disha is not financial advice.** It is a student project. Every screen carries
that warning and it is meant literally.

---

## Why "UP, DOWN or NEUTRAL" and not a price

Predicting an exact price is nearly impossible and, for most purposes, not even
useful. Predicting a direction is a simpler question that can at least be
scored: tomorrow either went up or it did not, so you can count how often the
guess was right.

**Why NEUTRAL exists.** Share prices almost never finish exactly flat — they
drift a few paise in some direction every day. If Disha only had UP and DOWN,
a day that moved +0.02% would count as a real "up" day, and the app would spend
its time predicting meaningless noise. So Disha draws a band around "no real
change":

| Instrument | Counts as NEUTRAL if tomorrow moves within |
|---|---|
| NIFTY 50 index | ±0.25% |
| A single stock | ±0.50% |

Anything inside the band is NEUTRAL. Anything above is UP, anything below is
DOWN. Single stocks get a wider band because an individual company bounces
around roughly twice as much as the whole index does.

---

## Starting the app

```
venv/Scripts/python.exe -m streamlit run dashboard/app.py
```

A browser tab opens showing the Disha terminal. The look is deliberate — green
text on black, ASCII-art boxes, scan lines, a halftone dot pattern in the
background. It is styled after an old trading terminal.

---

## The left sidebar — the control panel

Everything you can *change* lives in the sidebar on the left. If you do not see
it, click the small double-arrow at the top-left of the screen to slide it open.

### `> instrument` — which stock or index

A dropdown listing everything Disha knows about. Out of the box that is the
NIFTY 50 index plus seven large companies: TCS, Reliance, Infosys, HDFC Bank,
ICICI Bank, State Bank of India and ITC.

Each entry has a small badge in front of its name:

- **`[###]`** — all three prediction methods have been trained for this symbol.
  Ready to use.
- **`[##-]`** — two of three are ready.
- **`[---]`** — nothing has been trained yet. You will be offered a button to do
  it.

### `> model` — which prediction method

Disha has three different ways of making its guess, and you choose which one you
are looking at. They are genuinely different techniques, not settings on one
technique:

- **Random Forest** — imagine five hundred people each given a few of
  yesterday's measurements and asked to vote on tomorrow. The majority vote is
  the answer. Simple, hard to fool, and surprisingly hard to beat.
- **XGBoost** — a chain of small guessers where each one tries to fix the
  mistakes the previous ones made. Usually the strongest of the three on this
  kind of table-shaped data.
- **LSTM** — a neural network that reads the last 30 trading days *in order*,
  like a sentence, so it can in principle notice patterns that unfold over
  several weeks. It is by far the most complicated of the three, and on this
  problem it performs the worst. That result is itself a finding.

The dropdown only lists methods that have actually been trained. If one is
missing, a line underneath tells you which.

### `+ add another symbol` — adding something that is not in the list

Click to expand, then fill in three things:

| Field | What to put | Example |
|---|---|---|
| **yahoo symbol** | The code Yahoo Finance uses | `WIPRO.NS` |
| **display name** | Any name you want to see in the list | `Wipro` |
| **kind** | `stock` for one company, `index` for a basket | `stock` |

NSE-listed companies end in `.NS`. Indices start with `^` — for example
`^NSEBANK` for the Bank Nifty.

Press **ADD + TRAIN**. Disha downloads the full price history, computes
everything it needs, and trains all three methods. This takes roughly three to
six minutes, during which you get the loading screen described below. When it
finishes, your new symbol is in the dropdown permanently — it is saved to disk
and will still be there next time you open the app.

If you enter a symbol Yahoo does not recognise, you get a clear error message
rather than a crash.

### The `DATA` box — is this up to date?

A small box showing the date of the most recent trading day Disha has, and one
of two statuses:

- **`CURRENT`** — the data is fresh.
- **`N SESSIONS BEHIND`** — the market has traded N days since Disha last
  downloaded anything. The box turns amber as a nudge.

### The two buttons

**`REFRESH MARKET DATA`** — downloads the latest prices, recomputes all the
measurements, and re-runs the already-trained methods against the new data. Use
this day to day. Takes well under a minute. It does **not** retrain anything.

**`RETRAIN ALL MODELS`** — throws away the trained methods and rebuilds them
from scratch, running the full testing study described later. Takes a few
minutes. You only need this if you have changed something in the code or want to
re-verify the results.

### The `INSTRUMENT` box

Reference information about whatever is currently selected: its Yahoo symbol,
whether it is a stock or an index, the NEUTRAL band being used for it, and
whether it is one of the built-ins or something you added yourself.

---

## The loading screen — meet Disha

Whenever something slow happens — adding a symbol, refreshing data, retraining —
the screen clears and **Disha** appears. She is the app's mascot, drawn entirely
in keyboard characters, and she is there for a practical reason: training takes
minutes, and a frozen screen with no explanation makes people think the app has
crashed.

While you wait, you get:

- **Disha's face**, which animates — she blinks and changes expression, so you
  can see at a glance that things are still moving.
- **A progress bar** with a real percentage. This is not decoration. The
  training process announces each stage as it completes, and Disha counts those
  announcements against the total number expected, so the bar reflects genuine
  progress.
- **A spinner and a rotating line of chatter** from Disha, telling you what is
  happening right now.
- **The last few lines of the actual log**, in dim text, for anyone who wants to
  watch the details.
- **`do not close this tab`**, blinking, because closing it mid-training leaves
  you with a half-finished job.

If something goes wrong, Disha does not pretend otherwise: the screen clears and
you get a red panel with the end of the error log, so you can see what failed.

---

## The main screen

### The top strip

Across the top: the **DISHA NIFTY** logo in large ASCII letters, then a status
line reading something like

```
SYS ONLINE │ SBIN.NS │ STOCK │ MODEL LSTM │ LAST BAR 16 OCT 2026 │ 2,648 SESSIONS
```

which confirms what you are looking at and how much history is loaded.

Below that, four quick numbers:

- **Last close** — the most recent closing price, with the day's change.
- **Bars loaded** — how many trading days of history Disha has.
- **RSI (14)** — a standard 0-to-100 gauge of how hard the price has been pushed
  recently. Above 70 is usually called "overbought", below 30 "oversold".
- **Volatility** — how jumpy the price has been over the last ten days, stated
  as a yearly figure. 15% is a calm stock; 45% is a nervous one.

Then a **sparkline** — the last 180 closing prices squeezed into a single line
of text characters, with the high and low labelled. A shape-at-a-glance, no
chart needed.

### Four tabs

Everything else is organised into **TODAY**, **TRACK RECORD**, **MODELS** and
**HOW IT WORKS**.

---

## Tab 1 — TODAY

The actual prediction.

### The call

A large panel showing one of three words:

- **`> BUY`** — the model expects UP.
- **`> SELL`** — the model expects DOWN.
- **`> HOLD`** — the model expects NEUTRAL, *or* it is not confident enough to
  commit.

Underneath, in smaller text: the direction it leaned and a **confidence**
percentage, then a one-line reason.

### The confidence gate — the most important feature on this screen

Because there are three possible answers, a model that knows nothing would still
score about 33% on each. So when a model says "UP, 36% confident", it has
technically chosen UP, but it is barely distinguishable from a shrug.

Disha therefore applies a rule: **below 45% confidence, the action becomes
HOLD** — regardless of which direction won. The direction it leaned is still
shown, so you can see what it thought, but the recommendation is "do nothing".

In practice this fires often. That is not a bug; it is Disha declining to
pretend. An app that always gave you a confident BUY or SELL would be lying to
you about how much it knows.

### The probability bars

To the right, three bars drawn in text characters:

```
UP      ███████████████·················  34%
NEUTRAL █████████████████···············  38%
DOWN    ████████████····················  28%
```

The three always add to 100%. A marker on the bars shows where the 45%
confidence line sits, so you can see immediately whether the winner cleared it.

Bars of roughly equal length mean the model genuinely cannot tell — which is the
normal case, and worth seeing rather than hiding.

### Most recent settled call

A small box showing the last prediction whose outcome is actually known: what
the model said, what the market did, the size of the move, and **HIT** or
**MISS**. The box turns red on a miss. No cherry-picking — it is always the most
recent one.

### Indicator context

An expandable section with plain-English notes about the current state of the
stock: whether RSI looks overbought or oversold, whether MACD is leaning bullish
or bearish, whether the price is above or below its 50-day average, and whether
volatility is high, low or ordinary.

**An honest caveat:** these notes describe the *information the model was given*.
They are not the model's reasons. A forest of five hundred decision trees does
not have a reason you can put into a sentence. Read them as background, not as
an explanation of the call.

### Charts

A **candlestick chart** of the recent price, with a 20-day and 50-day average
drawn over it. Each candle is one day: the body spans open to close, green if it
closed up, red if down, and the thin lines show the day's full range. You can
switch the window between 60, 120, 250 and 500 days.

Below it, the **RSI chart**, with dotted lines at 70 and 30.

---

## Tab 2 — TRACK RECORD

How the chosen method has actually performed, day by day. This is the tab to
visit if you are wondering whether to trust anything in TODAY.

Crucially, these are **genuine out-of-sample results**. The method producing
these calls was trained only on data up to the end of 2023, then shown 2024
onwards *once*, and never retrained on it. It had no opportunity to peek at the
answers.

### Four numbers

- **Accuracy** — how often it was right, with a comparison line underneath.
- **Calls** — how many predictions are in the window you selected.
- **When taking a side** — accuracy counting only the days it committed to UP or
  DOWN, ignoring all the NEUTRAL days. A different and more interesting
  question: when it does stick its neck out, is it right?
- **Sided calls** — how many days those were.

### The comparison line, and why it is not 33%

The accuracy box says something like `-6.6% vs always neutral`. That comparison
matters more than the accuracy itself.

The obvious benchmark is 33.3% — random guessing between three options. But
there is a much better one-line strategy available: **always say NEUTRAL**.
Since roughly 35% of days genuinely are flat, that trivial rule scores about
35% while thinking about nothing at all.

So 35%, not 33%, is the bar. A model scoring 34% is not "slightly better than
chance" — it is **worse than a strategy you could write on a napkin**. Disha
compares against that honest baseline everywhere, and when a model falls short
it says so with a minus sign.

### Each call against the outcome

A chart of the closing price with a marker on every prediction day:

- **Marker shape** is what the model predicted — triangle up for UP, triangle
  down for DOWN, circle for NEUTRAL.
- **Marker colour** is whether it worked — green for agreement, red for not.

Hover over any marker for the date, the price, the prediction with its
confidence, and the actual move. A quick way to spot whether the misses cluster
in particular market conditions.

You can switch the span between the last 30, 60, 120, 250 days or all of them.

### Rolling accuracy, all models

One line per method, each showing its accuracy over a moving 42-day window
(about two trading months), with the always-NEUTRAL baseline drawn as a dotted
horizontal line.

The lines cross back and forth over that dotted line. The caption says what that
means: **this is what a model without a durable edge looks like.** A method with
real predictive power would stay above the line most of the time. These wander.

### Recent calls

The last fifteen predictions as a plain table: date, closing price, what the
model said, how confident it was, what the market did, the actual percentage
move, and Hit or Miss in green or red.

---

## Tab 3 — MODELS

Compares all three methods head to head on the sealed test period.

### Accuracy bars

An ASCII bar chart with one bar per method, plus a bar for the always-NEUTRAL
baseline, so you can see at a glance whether any of them beat it.

Underneath is a line like *"one standard error is about 1.8%, so gaps smaller
than that are noise."* This is Disha stopping you from over-reading the chart. If
Random Forest scores 33.3% and the LSTM scores 31.9%, that 1.4-point gap is
smaller than the measurement error — the honest conclusion is that they are
tied, not that one is better. Anyone who tells you a model is better by half a
percent is reading tea leaves.

### The results table

Per method:

- **Accuracy** — how often the direction was right.
- **Macro F1** — a fairer score than accuracy when one answer is much more
  common than the others. A model that always says NEUTRAL gets decent accuracy
  but a poor macro F1, which exposes it.
- **Sharpe** — return earned per unit of nerve-wracking. Higher is better;
  below 1 is usually considered unimpressive.
- **CAGR** — the yearly growth rate the strategy would have achieved.
- **Max drawdown** — the worst peak-to-trough fall along the way. The number
  that tells you whether you could have actually sat through it.
- **Trades** — how many times the strategy changed its position. More trades
  means more cost.

### Equity curve

What would have happened to ₹1 from the start of 2024 if you had followed each
method, plus a dashed line for simply buying and holding.

Two things to know about this chart:

- **Trading costs are charged.** Every change of position pays 0.05%. This
  matters enormously: a method that flips between long and short every day looks
  wonderful until you subtract brokerage, at which point it usually looks
  terrible. Disha subtracts it.
- **Buy and hold is a serious competitor.** Beating it is the real test, and it
  does not require any model at all.

### Confusion matrix

A grid showing exactly which answers got confused with which:

```
actual \ pred       DOWN  NEUTRAL       UP     row %
──────────────────────────────────────────────────────
DOWN                  41       98       72     19.4%
NEUTRAL               55      142       81     51.1%
UP                    48      104       65     29.9%
```

Read along a row: of all the days that genuinely were DOWN, how many did the
model call DOWN, NEUTRAL and UP? The **row %** is how often it got that
particular answer right.

This tells you things accuracy alone hides. A model might score respectably
overall while being completely blind to DOWN days — valuable to know, and
visible here in one glance.

### Per-fold results

An expandable section with the year-by-year testing results described next.

---

## Tab 4 — HOW IT WORKS

A diagram and short write-up inside the app covering the same ground as this
document in condensed form: where the data comes from, how the measurements are
built, how the testing is structured, and why there are two copies of every
model.

---

## What happens when Disha trains — in plain words

This is the part that matters most, because it is what separates Disha from the
many apps that claim stock predictions and quietly cheat.

### Step 1 — Get the prices

Download the daily open, high, low, close and volume from Yahoo Finance, from
January 2015 to today. Also download two background series: the India VIX (a
"fear gauge") and the rupee-dollar exchange rate. For a single company, the
NIFTY 50 is downloaded too, so the model can see what the wider market was
doing on the same day.

Prices are adjusted for splits. Without that adjustment, the day TCS split its
shares would look like a 50% crash, and the model would dutifully learn that one
specific date in history was a disaster — which teaches it nothing about the
future.

### Step 2 — Turn prices into measurements

About 26 measurements (28 for a single stock) are computed for each day: recent
returns, averages over different periods, momentum, volatility, RSI, MACD,
Bollinger Bands, volume relative to normal, and so on.

**All of them are ratios or percentages, never a rupee price.** This sounds like
a detail but it is central. If the model learned a rule like "sell when the
50-day average is below ₹9,000", that rule was useful in 2015 and is nonsense
today, when the same thing trades at ₹24,000. Ratios — "price is 3% above its
50-day average" — mean the same thing in every year, which is the only way a
decade of history can be used together.

### Step 3 — Write down the right answer

For each day, look at what happened the next day and label it UP, NEUTRAL or
DOWN using the band described earlier. The very last day gets thrown away,
because tomorrow has not happened yet and so there is no answer to record.

### Step 4 — Test it the way time actually works

This is where most amateur attempts go wrong, and where Disha is deliberately
strict.

The tempting shortcut is to shuffle all ten years, train on a random 80% and
test on the other 20%. For a time series that is **cheating**: the model ends up
trained on 2024 and tested on 2019, which means it has already seen the future
it is being asked to predict. Results from that approach look fantastic and mean
nothing.

Disha instead walks forward through time, exactly as a real forecaster would:

| Round | Learns from | Then guesses | That year was |
|---|---|---|---|
| 1 | 2015–2019 | 2020 | the COVID crash |
| 2 | 2015–2020 | 2021 | the post-COVID boom |
| 3 | 2015–2021 | 2022 | the rate-hike slump |
| 4 | 2015–2022 | 2023 | a calm, stable year |

Each round only ever learns from the past. And each test year is a deliberately
different kind of market, so a model that only works when everything is going up
gets caught out four separate times.

### Step 5 — The sealed final exam

Everything from January 2024 onwards is locked away and used **once**. The model
is trained on 2015–2023, shown the sealed period a single time, and the score is
recorded. That one number is the only honest accuracy figure in the app, and it
is what MODELS and TRACK RECORD display.

It is used once because repeated testing is a slow form of cheating: look at the
same test set twenty times while tweaking settings, and you will eventually find
settings that happen to suit that particular stretch of history. That is not
skill, it is coincidence dressed up as a result.

### Step 6 — Two copies of every model

When the exam is over, Disha saves the examined model and then trains a **second
copy on absolutely everything**, including the sealed period.

| | The examined copy | The live copy |
|---|---|---|
| Learned from | 2015–2023 | all of it, up to yesterday |
| Was it scored? | yes, once | never |
| Used for | every accuracy number you see | tomorrow's call in TODAY |

The reason for the split: the examined copy gives you an honest score but is a
couple of years out of date. The live copy knows about last week, which is what
you want for tomorrow's prediction — but because it has seen everything, any
accuracy you computed for it would be meaningless, like grading an exam the
student had already been given the answers to.

Mixing these two up is the single most common way projects of this kind end up
announcing an edge that was never there. Disha keeps them in separate files with
different names so it cannot happen by accident.

---

## What happens when you just click around

Normal use is fast because all the slow work was done at training time. Changing
the stock or the method means Disha reads some saved files, loads the trained
model, and runs one calculation. No downloading, no training.

Results are remembered so that clicking the same thing twice does not redo the
work. The clever part is *when* that memory is thrown away: Disha watches the
timestamps on the data file and the model file, and if either changes — because
you refreshed or retrained — the remembered answer is discarded automatically.
You can never be shown yesterday's prediction after a retrain.

---

## The honest finding

The app says this itself, on the HOW IT WORKS tab, and it deserves repeating:

> **Across both the index and individual stocks, predicting tomorrow's direction
> from price history alone performs within about one standard error of simply
> guessing the most common answer every day.**

In plain terms: after all of this — ten years of data, twenty-eight
measurements, three different techniques including a neural network, careful
testing across four market regimes — none of it reliably beats a person who says
"probably flat" every single morning.

Nor did complexity help. The LSTM, by far the most sophisticated of the three,
is the **worst** performer. That is a genuinely useful result, and the opposite
of what most people expect.

**Why this is a good outcome for the project.** Any version of this app that
claimed 70% accuracy would be wrong, and the error would be hiding in the
testing method — a shuffled split, a peek at the test set, a baseline of 33%
instead of 35%, or a backtest with no trading costs. Disha closes all of those
doors, and what is left on the other side is the truth: daily stock direction is
extremely close to unpredictable from price history alone.

What the project demonstrates, then, is not a profitable signal. It is the
method — correct handling of time, a sealed test set, an honest baseline, real
trading costs, a confidence gate, and the discipline to report a negative result
instead of hunting for a flattering one.

Treat every prediction Disha makes as a demonstration of that method, and
nothing more.

---

## Credits

Disha was developed as part of our semester by:

| Name | Roll number |
|---|---|
| Arindam Pal | BCA/40002/24 |
| Sujal Kumar | BCA/40005/24 |
| Khushi Kumari | BCA/40016/24 |
| Sneha Tiwari | BCA/40020/24 |

The same credits appear in the footer at the bottom of every screen in the app.

> Educational project output, not investment advice. Predictions are statistical
> estimates and are frequently wrong.
