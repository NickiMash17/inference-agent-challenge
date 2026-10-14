# Part 2 seed data — Atlas AI Summit RSVP inbox + calendar

This file is ready-to-paste content for manual Gmail / Google Calendar
seeding. It is **not** used by any Part 1 code and is not read by any
script in this repo — it exists purely as fixture content for the Part 2
(n8n) agent, which is built and documented separately.

**Speaker directory (ground truth):**
https://raw.githubusercontent.com/NickiMash17/inference-agent-challenge/main/data/kb/speakers.json

**Identity note:** every email below will actually be sent from one Gmail
account (mine) to `nene171408+atlas@gmail.com`, so the Gmail **From**
header will always be the same address for all 9 messages. It carries no
identity information. Each attendee's real identity — name, organisation,
email, phone — comes **only from the email body / signature text**, never
from the From header. The Part 2 agent must be built on that assumption.

There are **9 emails**, not 8, because one of the 8 planted errors (the
duplicate RSVP) inherently requires two messages — the original RSVP and
the later duplicate with a misspelled name. The other 7 planted errors
each live in their own single email. This is called out again at that
item below.

---

## 1. RSVP emails

### Email 1 of 9 — control case (clean, fully correct)

**Subject:** RSVP - Atlas AI Summit

**Body:**

```
Hi,

Confirming I'll be attending the Atlas AI Summit on 18-19 November.
Looking forward to it!

Best,
Troy Harris
Quantum Ridge Technologies
troy.harris@example.org
062 629 5209
```

*(Planted error: none — this is the control case. Name, organisation,
email, and phone all match the directory entry for SP-005 exactly, phone
reformatted with spaces but same digits.)*

---

### Email 2 of 9 — messy phone format

**Subject:** re: Atlas AI Summit attendance

**Body:**

```
Hey team

Yep I'm in for the summit, both days. Here are my details in case you
need them for badges:

Andrew Kane
Jacaranda Systems
andrew.kane@example.org
tel: 087 159 2445

See you there
```

*(Planted error: phone number in a messy local format. Directory phone
for SP-004... check — directory phone for Andrew Kane (SP-013) is
`0871592445`; this email gives the same digits as `087 159 2445`. Name,
org, and email all match the directory exactly.)*

---

### Email 3 of 9 — misspelled organisation name

**Subject:** RSVP: Atlas AI Summit

**Body:**

```
Hello,

I'd like to confirm my attendance at the Atlas AI Summit in November.

Thanks,
Latoya Wilson
Silver Line Robotics
latoya.wilson@example.com
```

*(Planted error: organisation misspelled as "Silver Line Robotics" —
directory organisation for SP-016 is "Silverline Robotics" (one word).
Name and email match the directory exactly; no phone given.)*

---

### Email 4 of 9 — missing email address

**Subject:** Atlas Summit - confirming attendance

**Body:**

```
Hi there,

Just confirming I'll be at the Atlas AI Summit this November. You can
reach me on 073-359-1551 if anything changes.

Regards,
Joseph Torres
Horizon Cognition
```

*(Planted error: no email address given anywhere in the body — only name,
organisation, and phone. Directory phone for SP-019 is `073-359-1551`,
matching exactly, so the missing email should be resolvable from the
directory via name+phone.)*

---

### Email 5 of 9 — Jon/John Mokoena identity collision

**Subject:** RSVP - Atlas AI Summit 2026

**Body:**

```
Hi,

I'll be attending the Atlas AI Summit on both days. My details:

John Mokoena
Nova Analytics
john.mokoena@example.org
083 400 0938

Thanks,
John
```

*(Planted error: this is a genuine identity collision, not a typo with an
obvious fix. The directory has two separate speakers who collide here:
SP-001 "Jon Mokoena" / Nova Analytics / jon.mokoena@example.org /
0783400938, and SP-002 "John Mokoena" / Ironwood AI Research /
john.mokoena@example.com / 0637548524. This email's **name spelling**
("John") matches SP-002 exactly, but its **organisation** ("Nova
Analytics") matches SP-001 exactly, and its email/phone match neither
exactly (close to both but not identical — `john.mokoena@example.org` is
SP-002's local part at SP-001's domain-style TLD, and the phone
`0834000938` isn't a reformat of either speaker's number). This should be
flagged for human review, not silently resolved to either speaker ID.)*

---

### Email 6 of 9 — email differs from directory

**Subject:** Re: Atlas AI Summit — RSVP

**Body:**

```
Hi,

Confirming attendance for the Atlas AI Summit. My contact info:

Leslie Frey
Khanya AI Labs
lesliefrey@example.com
084-595-7566

Cheers
```

*(Planted error: email address `lesliefrey@example.com` differs from the
directory's `leslie.frey@example.org` for SP-011 (different local-part
formatting and different domain). Name, organisation, and phone all match
the directory exactly, so the identity match is strong even though the
email itself doesn't match.)*

---

### Email 7 of 9 — person not in the directory at all

**Subject:** Atlas AI Summit - RSVP

**Body:**

```
Hi there,

I saw the summit announcement and would like to attend. Not a speaker,
just keen to attend as a delegate.

Priya Naidoo
DataForge Analytics
priya.naidoo@example.com
082 310 4456
```

*(Planted error: "Priya Naidoo" does not appear anywhere in
`speakers.json`. This is a legitimate new attendee, not a data error to
correct — the correct handling is to add her to the attendee list as a
non-speaker delegate, not to try to match her to an existing speaker
record.)*

---

### Email 8 of 9 — duplicate RSVP, part 1 of 2 (original)

**Subject:** RSVP - Atlas AI Summit

**Body:**

```
Hi,

Confirming I'll attend the Atlas AI Summit on 18-19 November.

Paige Williams
Nova Analytics
paige.williams@example.com
+27643819886
```

*(No error on its own — this is the first, correct RSVP from SP-008. It
exists so Email 9 below has something to duplicate.)*

---

### Email 9 of 9 — duplicate RSVP, part 2 of 2 (same person, name typo)

**Subject:** Atlas AI Summit - RSVP (sorry, resending)

**Body:**

```
Hi again,

Sorry, I don't think my earlier RSVP went through properly — resending.

Paige Willams
Nova Analytics
paige.williams@example.com
+27643819886

Thanks!
```

*(Planted error: the same person RSVPs twice. Email and phone are
identical to Email 8, organisation matches, but the name is misspelled
"Paige Willams" (missing an "i"). These two emails should be deduplicated
into a single attendee, SP-008, canonical name "Paige Williams".)*

---

## 2. Google Calendar events

### Event 1

- **Title:** Atlas AI Summit — Welcome Reception
- **Date/time:** 18 November 2026, 18:00–20:00 (Hall A foyer)
- **Description:**

```
Welcome reception for all Atlas AI Summit attendees and speakers.

Confirmed attendees include:
- Troy Harris, Quantum Ridge Technologies
- Latoya Wilson, Silverline Robotics
- Paige Williams, Nova Analytics
```

*(No planted error in this event. All three names/organisations here
match the directory exactly and overlap with Emails 1, 3, 8/9 above —
useful for cross-checking that the agent treats matching calendar and
email data as corroborating, not conflicting.)*

### Event 2

- **Title:** Atlas AI Summit — Closing Networking Lunch
- **Date/time:** 19 November 2026, 12:30–13:30 (Hall B)
- **Description:**

```
Closing lunch, open to all attendees.

Confirmed attendees include:
- Andrew Kane, Jacaranda Systems
- Leslie Frey, Khanya AI Labs
- Joseph Torres, Horizon Cognitions
```

*(Planted error — the one calendar error called for in the brief:
"Horizon Cognitions" has an extra trailing "s". The directory
organisation for SP-019 Joseph Torres is "Horizon Cognition". This
overlaps with Email 4 above, where the same person's organisation was
given correctly — a good test of using the email RSVP to correct the
calendar typo, or vice versa flagging the mismatch.)*

---

See `data/synthetic/part2_answer_key.json` for the full list of planted
errors (source, field, planted value, correct value/resolution) and the
expected final deduplicated attendee list.
