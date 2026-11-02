# Adversarial, Property-Based, Metamorphic and Release Test Matrix

## Test philosophy

The following tests target *offline validators, dataset-cleaning code, training preparation and model diagnostics*. They are not a license to use rule-based functions at final inference. For generative model tests, score raw text and count failures; do not rewrite it.

Each test records: `test_id, input/fixture, expected invariant, layer, observed result, pass/fail, provenance`. Expected labels for synthetic tests derive only from the schema; never from dev/test or another model. Run parser/cleaner unit tests on CPU; reserve GPU for model tests.

## A. Schema and JSON (12 checks)

| ID | Case | Assertion |
|---|---|---|
| A01 | Exactly eight keys | Accept valid schema |
| A02 | Missing `priority` | Reject |
| A03 | Extra `explanation` | Exact-match failure |
| A04 | `"amount_paise":"650000"` | Reject wrong type |
| A05 | `"needs_human":1` | Reject integer masquerading as bool |
| A06 | `"needs_human":"true"` | Reject string |
| A07 | `"category":"payments"` | Reject enum |
| A08 | Uppercase `"P1"` vs lowercase `"p1"` | Only canonical accepted |
| A09 | Array instead of object | Reject |
| A10 | Markdown fenced JSON | Raw generation invalid |
| A11 | Leading/trailing whitespace | Allowed strip only |
| A12 | Multiple JSON objects or trailing explanation | Invalid exact output |

## B. Amount and thresholds (14 checks)

| ID | Input evidence | Expected amount/priority condition |
|---|---|---|
| B01 | ₹6,500 failed payment | 650000 paise, P2 |
| B02 | ₹50,000 failed payment | 5000000 paise, P1 |
| B03 | ₹49,999.99 refund | 4999999 paise, P2 |
| B04 | ₹5,000 refund | 500000 paise, P2 |
| B05 | ₹4,999.99 refund | 499999 paise, P3 absent other trigger |
| B06 | ₹1.30k | 130000 paise |
| B07 | ₹1.5 lakh | 15000000 paise, P1 if disputed |
| B08 | ₹2,74,981 | 27498100 paise |
| B09 | Rs 21,378.99 | 2137899 paise |
| B10 | Complaint ₹300, balance ₹99,999 | amount 30000 paise; ignore balance |
| B11 | Complaint ₹300, daily limit ₹50,000 | amount 30000; don't P1 from limit |
| B12 | Fee ₹30, claim ₹5,100 | amount 510000, ignore fee |
| B13 | Amount absent | null |
| B14 | More than 2 decimal places/unclear currency | Flag train-label ambiguity; no guess |

## C. Categories, priority and human escalation (14 checks)

| ID | Ticket evidence | Expected |
|---|---|---|
| C01 | 'payment I did not authorize' + 'refund' | fraud wins, P1, human true |
| C02 | Unauthorized withdrawal ₹100 | fraud, P1, human true |
| C03 | Failed payment ₹100 | payment_failed, P3 |
| C04 | Duplicate charge ₹100 | refund, P3 |
| C05 | KYC pending with ₹40k balance | kyc, P3, txn fields null |
| C06 | Password reset with card amount mentioned | account_access, P2, txn fields null |
| C07 | 'I will report to RBI' on offers ticket | P1, needs_human true |
| C08 | 'consumer court' on ₹200 refund | P1, human true |
| C09 | 'URGENT!!' without legal or repeat | Priority unchanged |
| C10 | Third time contacting support on P3 offers | needs_human true; priority stays P3 |
| C11 | 'baar baar' repeat on P3 ticket | needs_human true |
| C12 | 'Pehle bhi complaint kar chuka hoon' | needs_human true |
| C13 | Ticket number only, no clear same-issue repeat | Do not infer human escalation automatically |
| C14 | Fraud with `needs_human=false` label | Strong invariant violation |

## D. Dates, references and channels (16 checks)

| ID | Ticket evidence | Expected |
|---|---|---|
| D01 | Received 2026-09-14, yesterday | 2026-09-13 |
| D02 | Received 2026-01-01, yesterday | 2025-12-31 |
| D03 | Received 2026-03-01, 2 days ago | 2026-02-27 |
| D04 | Received 2028-03-01, parso | 2028-02-28 (leap year) |
| D05 | Received 2026-01-02, Dec 31 | 2025-12-31 |
| D06 | Received 2026-08-10, Aug 9 | 2026-08-09 |
| D07 | Quoted `> On Aug 3` with no user transaction date | null |
| D08 | Quoted date Aug 3 plus user 'yesterday' on Aug 10 | 2026-08-09 |
| D09 | `np 12345 67890` | `NP1234567890` |
| D10 | `NP-12345-67890` | `NP1234567890` |
| D11 | Bank reference 123456789012 | txn_id null |
| D12 | Order #1234567890 | txn_id null |
| D13 | Ticket #12345 | txn_id null |
| D14 | Visa / debit card transaction | channel card |
| D15 | 'internet banking' transaction | channel netbanking |
| D16 | Source app-chat but payment channel not specified | channel null |

## E. Language and distraction (10 checks)

| ID | Ticket evidence | Expected |
|---|---|---|
| E01 | 'Bhai, paise kat gaye' | hi-en |
| E02 | 'Refund nahi aaya' | hi-en |
| E03 | English message from 'Rahul Sharma' | en |
| E04 | English ticket with Hindi user name in signature | en |
| E05 | English main message plus quoted boilerplate | en, no quote interference |
| E06 | Hinglish with English technical words | hi-en |
| E07 | Misspelled 'reawrd' on cashback | offers semantics remain |
| E08 | Phone model `Redmi Note 12` | Don't extract 12 as amount or txn ID |
| E09 | App version `6.3.7` | Don't extract as amount/date |
| E10 | Quoted support email includes dates and ticket IDs | Ignore quoted dates as txn dates; avoid ref hallucination |

## F. Metamorphic/property tests (12 checks)

| ID | Transformation | Invariant |
|---|---|---|
| F01 | Add unrelated account balance | relevant amount unchanged |
| F02 | Add unrelated daily limit | relevant amount and priority unchanged |
| F03 | Swap refund ₹4,999.99 -> ₹5,000 | P3 -> P2 (absent other triggers) |
| F04 | Swap refund ₹49,999.99 -> ₹50,000 | P2 -> P1, needs_human true |
| F05 | Add explicit legal threat | priority P1, human true, category unchanged absent fraud |
| F06 | Add explicit prior support contact | human false -> true, priority unchanged |
| F07 | Add unrecognized unauthorized transaction | fraud category wins |
| F08 | Change receipt date by +1 day keeping 'yesterday' | txn date +1 day |
| F09 | Change whitespace/dashes/case inside NP ID | canonical ID unchanged |
| F10 | Add quoted historical email lines | active transaction date unchanged |
| F11 | Change source email -> twitter leaving ticket semantics | classification unchanged unless relevant evidence changed |
| F12 | Run cleaner twice on its own output | identical labels and no second action log entries |

## G. System integration and failure injection (12 checks)

| ID | Failure | Expected system reaction |
|---|---|---|
| G01 | Invalid ZIP path traversal member | Extraction rejected |
| G02 | Invalid UTF-8 JSONL line | Explicit failure with line number |
| G03 | Duplicate train id with conflicting content | Halt audit; do not choose arbitrarily |
| G04 | Missing cleaning log id for fixed row | Release gate fails |
| G05 | Missing labels after SFT tokenization | Training fails before GPU run |
| G06 | Fully truncated assistant target | Training fails before GPU run |
| G07 | Requested trainable params >50M | Training fails |
| G08 | Injected NaN loss | Run aborts, manifest marks failed |
| G09 | Simulated OOM | Preserve last valid checkpoint; no false completion |
| G10 | Interrupted prediction write | `.partial` never treated as final |
| G11 | Duplicate/missing test prediction id | Release gate fails without rewriting output |
| G12 | Reload adapter under mismatched base revision | Release gate fails |

## H. Compliance and research integrity (10 checks)

| ID | Test | Expected |
|---|---|---|
| H01 | Search training call graph for dev/test labels | None |
| H02 | Search generation call graph for evidence repair, rules, APIs | None |
| H03 | Compare training prompt and fixed inference messages | No additional rules/examples in final |
| H04 | Verify no JSON grammar/constrained decoder | None |
| H05 | Verify test strings are direct decode+strip | Raw immutable output |
| H06 | Re-score dev using unmodified supplied scorer | Identical official metrics |
| H07 | Fresh T4 top-to-bottom timing | Under 180 minutes measured |
| H08 | New-process PEFT adapter replay | Actual reproducibility evidence |
| H09 | Inspect staged Git files | No raw corpus, base weights, secrets |
| H10 | Verify report numbers map to log artifacts | No unsupported result |

## Model-level caveat

A trained neural model can fail any semantic invariant in these tests. A test failure is a measurement requiring an **offline** training/data fix and re-evaluation. It does not permit an inference-time retry or patch. A schema validator may fail release if predictions are invalid; it cannot make them valid without generating compliant model outputs from an improved model.

## Coverage requirements

- Test at least one case per output field and each P1/P2 boundary; exercise both English and Hinglish, all four sources, long quoted cases, null fields, false NP ID, `k/lakh`, and year rollover.
- Use parameterized tests for all known spelling/format variants and separate deterministic unit-test oracle values from actual model-generated predictions.
- Property-based testing may explore legal numeric/date/reference ranges, but its oracle may only derive from `SCHEMA.md`; it cannot create training labels with an external model.
- Distinguish schema validator pass rates from raw model exact-match metrics.

