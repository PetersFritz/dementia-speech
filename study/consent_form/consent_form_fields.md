# Consent Google Form — field-by-field spec

Build this as a single Google Form. Enable **Settings → Responses → Collect
email addresses** (required — `onSubmit.gs` depends on it). Suggested
structure, in order:

---

**Form title:** [Study Title] — Information Sheet & Consent

**Form description:**
> Please read the information below carefully before deciding whether to
> take part. If you consent and submit this form, you will automatically
> receive an email with your personal link to the study.

---

### Section 1 — Information Sheet

**Type:** Section header, or a "Paragraph" content block

Paste the full contents of `information_sheet.md` (with placeholders
filled in) here, or link to it if hosted elsewhere (e.g. a PDF on your
department site). Google Forms doesn't render Markdown, so convert `##`
headings to bold text and strip the `#`/`*` syntax when pasting.

---

### Section 2 — Consent

Each of the following as a **required** question, type **"Multiple
choice"** with a single option **"Yes"** (forces an active tick rather than
an assumed default — standard practice for consent forms), or as a
**checkbox grid** if you prefer them grouped:

1. *"I confirm that I have read and understood the information sheet above
   (version [X], dated [DATE]) and have had the opportunity to ask
   questions."* — Yes
2. *"I understand that my participation is voluntary and that I am free to
   withdraw at any time without giving a reason, and without penalty."* —
   Yes
3. *"I understand that my responses will be recorded against an anonymous
   code, and that my email address (collected below) is stored separately
   and used only to send me my personal study link."* — Yes
4. *"I agree that my anonymised/pseudonymised responses may be used in
   research outputs such as publications, presentations, or a thesis."* —
   Yes
5. *"I agree to take part in this study."* — Yes

*(Adjust wording to match whatever your ethics committee's template
requires — some Sheffield departments have a standard consent statement
set; check with your supervisor/ethics reviewer before finalising.)*

---

### Section 3 — Contact details

- **Email address** — this is the field auto-collected via the "Collect
  email addresses" setting, so you don't need to add a separate text
  question for it. Google shows it automatically as the first field once
  that setting is on.
- Optional: *"Name"* (short answer, not required) if you want to be able to
  address the invitation email personally — `onSubmit.gs` doesn't currently
  use this, but could be extended to (e.g. `response.getItemResponses()` to
  pull a name field into the email greeting).

---

### After building it

1. Confirm **Collect email addresses** is on (Settings → Responses).
2. Link a response Sheet (Responses tab → green Sheets icon).
3. Add the Apps Script from `study/consent_form/onSubmit.gs` and its
   installable trigger — see `study/README.md` section 1, steps 4–6.
4. Submit a test response yourself and confirm the email arrives.
