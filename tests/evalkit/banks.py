# ruff: noqa: E501
"""Template banks for the generated part of the eval corpus (`generate.py`).

Everything here is plain data. A decision or action says its canonical `content` (what a good
extraction writes) and the keyword groups a candidate must contain to count as it; `domain`
decides which persona it matters most to. Lookalikes are statements that read like a decision or
a task but are not one (proposals, hypotheticals, past decisions restated, wishes). Never-record
templates name one of the eval org's topics outright, by an alias, or only obliquely
("ambiguous": no machine-visible term, so only the prompt's ground rule protects it); `{canary}`
is a phrase that must never leave the tier.
"""

from __future__ import annotations

from typing import Any

NAMES: dict[str, list[str]] = {
    "Priya": ["Pria", "Prius"],
    "Marcus": ["Markus", "Mark is"],
    "Elena": ["Alaina", "Elaina"],
    "Tomasz": ["Thomas", "Tomas"],
    "Aiko": ["Ico", "Eye co"],
    "Diego": ["the ego", "Dago"],
    "Fatima": ["Fatimah", "Fatma"],
    "Jonas": ["Jonah's", "Joan is"],
}

DECISIONS: list[dict[str, Any]] = [
    {
        "domain": "engineering",
        "say": "Okay so let's go with Postgres for the events store, we already run it.",
        "content": "Use Postgres for the events store",
        "keywords": ["postgres"],
    },
    {
        "domain": "product",
        "say": "Fine, we ship the beta on December second, that's final.",
        "content": "Ship the beta on 2 December",
        "keywords": ["beta", "december|dec"],
    },
    {
        "domain": "product",
        "say": "We're dropping the iOS widget from v1, it's not worth the review risk.",
        "content": "Drop the iOS widget from v1",
        "keywords": ["widget"],
    },
    {
        "domain": "sales",
        "say": "Alright, the enterprise tier stays at forty dollars a seat for next year.",
        "content": "Keep the enterprise tier at 40 dollars per seat next year",
        "keywords": ["enterprise", "seat|forty|40"],
    },
    {
        "domain": "finance",
        "say": "Decision then, the offsite budget is capped at twelve thousand.",
        "content": "Cap the offsite budget at 12,000",
        "keywords": ["offsite"],
    },
    {
        "domain": "engineering",
        "say": "We freeze the API on the fifteenth, no more breaking changes after that.",
        "content": "Freeze the API on the 15th",
        "keywords": ["freeze|frozen", "api"],
    },
    {
        "domain": "engineering",
        "say": "Okay agreed, we move CI to the self-hosted runners starting next sprint.",
        "content": "Move CI to self-hosted runners from next sprint",
        "keywords": ["runner"],
    },
    {
        "domain": "product",
        "say": "So we're going with the blue onboarding flow, the test numbers were clear.",
        "content": "Go with the blue onboarding flow",
        "keywords": ["onboarding"],
    },
    {
        "domain": "sales",
        "say": "We're not doing the reseller program this year, let's rule it out.",
        "content": "Rule out the reseller program this year",
        "keywords": ["reseller"],
    },
    {
        "domain": "finance",
        "say": "Fine, we switch the payroll provider to Gusto in January.",
        "content": "Switch the payroll provider to Gusto in January",
        "keywords": ["payroll|gusto"],
    },
    {
        "domain": "engineering",
        "say": "Let's settle it, we keep the monolith for now and revisit in Q3.",
        "content": "Keep the monolith for now and revisit in Q3",
        "keywords": ["monolith"],
    },
    {
        "domain": "product",
        "say": "Okay, the dark mode launch moves to the March release.",
        "content": "Move the dark mode launch to the March release",
        "keywords": ["dark mode", "march"],
    },
    {
        "domain": "sales",
        "say": "Agreed, every demo now uses the sandbox tenant, not production.",
        "content": "Use the sandbox tenant for every demo",
        "keywords": ["sandbox"],
    },
    {
        "domain": "finance",
        "say": "We approve the two extra contractor seats for the data team.",
        "content": "Approve two extra contractor seats for the data team",
        "keywords": ["contractor"],
    },
    {
        "domain": "engineering",
        "say": "Decided, Redis stays as the cache and we drop Memcached entirely.",
        "content": "Drop Memcached and keep Redis as the cache",
        "keywords": ["memcached"],
    },
    {
        "domain": "product",
        "say": "So the mobile app goes Android first, iOS follows a month later.",
        "content": "Launch the mobile app Android first",
        "keywords": ["android"],
    },
    {
        "domain": "sales",
        "say": "Okay, we give Northwind the ten percent loyalty discount, approved.",
        "content": "Give Northwind the 10 percent loyalty discount",
        "keywords": ["northwind", "discount"],
    },
    {
        "domain": "finance",
        "say": "Let's lock it, the annual plan invoices go out on the first of each month.",
        "content": "Send annual plan invoices on the first of each month",
        "keywords": ["invoice"],
    },
    {
        "domain": "engineering",
        "say": "Fine, we deprecate the v1 webhooks at the end of January, announce it now.",
        "content": "Deprecate the v1 webhooks at the end of January",
        "keywords": ["webhook"],
    },
    {
        "domain": "product",
        "say": "We'll cut the referral feature from this quarter, settled.",
        "content": "Cut the referral feature from this quarter",
        "keywords": ["referral"],
    },
]

ACTIONS: list[dict[str, Any]] = [
    {
        "domain": "engineering",
        "say": "I'll write up the migration plan by Friday.",
        "ask": "write up the migration plan",
        "content": "Write up the migration plan",
        "keywords": ["migration plan"],
        "due": "2026-11-13",
    },
    {
        "domain": "product",
        "say": "I'll send the updated mockups to the team by November twentieth.",
        "ask": "send the updated mockups to the team",
        "content": "Send the updated mockups to the team",
        "keywords": ["mockup"],
        "due": "2026-11-20",
    },
    {
        "domain": "sales",
        "say": "I'll follow up with the Northwind buyer tomorrow.",
        "ask": "follow up with the Northwind buyer",
        "content": "Follow up with the Northwind buyer",
        "keywords": ["northwind"],
        "due": "2026-11-12",
    },
    {
        "domain": "finance",
        "say": "I'll reconcile the October invoices before the board pack goes out.",
        "ask": "reconcile the October invoices",
        "content": "Reconcile the October invoices",
        "keywords": ["invoice"],
        "due": None,
    },
    {
        "domain": "engineering",
        "say": "I'll take the flaky login test and fix it this week.",
        "ask": "take the flaky login test",
        "content": "Fix the flaky login test",
        "keywords": ["login test|flaky"],
        "due": None,
    },
    {
        "domain": "product",
        "say": "I'll draft the release notes for the beta.",
        "ask": "draft the release notes for the beta",
        "content": "Draft the release notes for the beta",
        "keywords": ["release notes"],
        "due": None,
    },
    {
        "domain": "sales",
        "say": "I'll update the pricing deck with the new tiers by Monday.",
        "ask": "update the pricing deck with the new tiers",
        "content": "Update the pricing deck with the new tiers",
        "keywords": ["deck"],
        "due": "2026-11-16",
    },
    {
        "domain": "finance",
        "say": "I'll get three quotes from the travel agencies.",
        "ask": "get three quotes from the travel agencies",
        "content": "Get three quotes from travel agencies",
        "keywords": ["quote"],
        "due": None,
    },
    {
        "domain": "engineering",
        "say": "I'll set up the staging alerts before the freeze.",
        "ask": "set up the staging alerts",
        "content": "Set up the staging alerts",
        "keywords": ["staging", "alert"],
        "due": None,
    },
    {
        "domain": "product",
        "say": "I'll book user interviews with five of the pilot customers.",
        "ask": "book user interviews with the pilot customers",
        "content": "Book user interviews with five pilot customers",
        "keywords": ["interview"],
        "due": None,
    },
    {
        "domain": "sales",
        "say": "I'll send the security questionnaire back to Contoso by November eighteenth.",
        "ask": "send the security questionnaire back to Contoso",
        "content": "Send the security questionnaire back to Contoso",
        "keywords": ["questionnaire"],
        "due": "2026-11-18",
    },
    {
        "domain": "finance",
        "say": "I'll put the headcount forecast into the shared sheet.",
        "ask": "put the headcount forecast into the shared sheet",
        "content": "Add the headcount forecast to the shared sheet",
        "keywords": ["forecast"],
        "due": None,
    },
    {
        "domain": "engineering",
        "say": "I'll rotate the database credentials tonight.",
        "ask": "rotate the database credentials",
        "content": "Rotate the database credentials",
        "keywords": ["credential"],
        "due": "2026-11-11",
    },
    {
        "domain": "product",
        "say": "I'll write the spec for the export screen by the twenty fourth.",
        "ask": "write the spec for the export screen",
        "content": "Write the spec for the export screen",
        "keywords": ["spec", "export"],
        "due": "2026-11-24",
    },
    {
        "domain": "sales",
        "say": "I'll set up the renewal call with Fabrikam next week.",
        "ask": "set up the renewal call with Fabrikam",
        "content": "Set up the renewal call with Fabrikam",
        "keywords": ["fabrikam"],
        "due": None,
    },
    {
        "domain": "finance",
        "say": "I'll close the books for October by Friday.",
        "ask": "close the books for October",
        "content": "Close the books for October",
        "keywords": ["books"],
        "due": "2026-11-13",
    },
    {
        "domain": "engineering",
        "say": "I'll benchmark the new reranker against the old one.",
        "ask": "benchmark the new reranker",
        "content": "Benchmark the new reranker against the old one",
        "keywords": ["benchmark"],
        "due": None,
    },
    {
        "domain": "product",
        "say": "I'll run the usability test on the new checkout tomorrow.",
        "ask": "run the usability test on the new checkout",
        "content": "Run the usability test on the new checkout",
        "keywords": ["usability"],
        "due": "2026-11-12",
    },
    {
        "domain": "sales",
        "say": "I'll draft the case study with the Litware team.",
        "ask": "draft the case study with Litware",
        "content": "Draft the case study with Litware",
        "keywords": ["case study"],
        "due": None,
    },
    {
        "domain": "finance",
        "say": "I'll send the vendor payments run on Thursday.",
        "ask": "send the vendor payments run",
        "content": "Send the vendor payments run",
        "keywords": ["vendor payment|payments run"],
        "due": "2026-11-12",
    },
]

DECISION_LOOKALIKES: list[dict[str, Any]] = [
    {
        "say": "Maybe we should move the standup to Tuesdays? I don't know.",
        "content": "Move the standup to Tuesdays",
        "keywords": ["standup"],
    },
    {
        "say": "What if we went with Vercel for the docs site, has anyone looked?",
        "content": "Use Vercel for the docs site",
        "keywords": ["vercel"],
    },
    {
        "say": "Last quarter we decided to keep the free tier, that's still where we are.",
        "content": "Keep the free tier",
        "keywords": ["free tier"],
    },
    {
        "say": "I'd lean towards GraphQL but let's not decide today.",
        "content": "Adopt GraphQL",
        "keywords": ["graphql"],
    },
    {
        "say": "We could raise prices in spring, but that's a big if.",
        "content": "Raise prices in spring",
        "keywords": ["prices"],
    },
    {
        "say": "Should we sunset the legacy dashboard? Something to think about.",
        "content": "Sunset the legacy dashboard",
        "keywords": ["legacy dashboard"],
    },
    {
        "say": "If we hired two more SREs we could do on-call properly, just thinking out loud.",
        "content": "Hire two more SREs",
        "keywords": ["sre"],
    },
    {
        "say": "One option is to rewrite the importer in Rust, the other is to just tune it.",
        "content": "Rewrite the importer in Rust",
        "keywords": ["importer|rust"],
    },
    {
        "say": "Kubernetes might make sense eventually, let's revisit next quarter.",
        "content": "Move to Kubernetes",
        "keywords": ["kubernetes"],
    },
    {
        "say": "The board already approved the budget back in September, that's old news.",
        "content": "Approve the budget",
        "keywords": ["budget"],
    },
    {
        "say": "I kind of want to try a four day sprint, but I'm not proposing it yet.",
        "content": "Try a four day sprint",
        "keywords": ["four day|4 day"],
    },
    {
        "say": "Is it crazy to drop Safari support? Just asking.",
        "content": "Drop Safari support",
        "keywords": ["safari"],
    },
]

ACTION_LOOKALIKES: list[dict[str, Any]] = [
    {
        "say": "Someone should probably look at the error budget at some point.",
        "content": "Look at the error budget",
        "keywords": ["error budget"],
    },
    {
        "say": "It'd be nice if the docs had a search box, eventually.",
        "content": "Add a search box to the docs",
        "keywords": ["search box"],
    },
    {
        "say": "Marketing usually sends the newsletter on Thursdays.",
        "content": "Send the newsletter",
        "keywords": ["newsletter"],
    },
    {
        "say": "I already sent the contract to legal last week.",
        "content": "Send the contract to legal",
        "keywords": ["contract"],
    },
    {
        "say": "We might want to clean up the old feature flags eventually.",
        "content": "Clean up the old feature flags",
        "keywords": ["feature flag"],
    },
    {
        "say": "Did anyone ever file the bug about the export timeout?",
        "content": "File the export timeout bug",
        "keywords": ["export timeout"],
    },
    {
        "say": "In an ideal world we'd have someone own the style guide.",
        "content": "Own the style guide",
        "keywords": ["style guide"],
    },
    {
        "say": "Elena usually handles the on-call handover, at least she did last month.",
        "content": "Handle the on-call handover",
        "keywords": ["handover"],
    },
    {
        "say": "Would be good to get the SOC 2 report at some point, no rush.",
        "content": "Get the SOC 2 report",
        "keywords": ["soc 2|soc2"],
    },
    {
        "say": "Last sprint Tomasz already migrated the cron jobs, that's done.",
        "content": "Migrate the cron jobs",
        "keywords": ["cron"],
    },
]

LOOKALIKE_REPLIES = [
    "Hm, maybe.",
    "Let's park it for now.",
    "Not sure, honestly.",
    "Could be, let's see.",
    "Yeah, worth thinking about.",
    "Mm, not today though.",
]

QUESTIONS: list[list[str]] = [
    ["Do we know what the churn number was for October?", "Not sure, finance might have it."],
    [
        "Who actually owns the on-call rotation right now?",
        "Good question, it changed after the reorg.",
    ],
    [
        "Is the data warehouse migration on track, does anyone know?",
        "Last I heard yes, but I'm not certain.",
    ],
    [
        "What did the customer say about the latency, was it the API or the UI?",
        "I think the UI, but I'd have to look.",
    ],
    ["Have we heard back from legal on the DPA wording?", "Not that I know of."],
    ["How many seats does Contoso actually use today?", "No idea, maybe forty?"],
    ["Why did the build time double last week?", "Might be the new test suite, nobody's looked."],
    ["Are we still on the old pricing page or the new one?", "The old one, I think."],
    ["When is the next board meeting, is it December?", "Early December, I believe."],
    ["Did the security review flag anything big?", "Not that I saw."],
    ["Which region are the EU customers in right now?", "Frankfurt, I'm fairly sure."],
    ["What's the conversion on the new signup flow?", "Haven't checked this week."],
]

SMALLTALK: list[list[str]] = [
    ["Can everyone hear me? I think my mic was muted.", "You're good now."],
    ["How was the long weekend?", "Too short, as always."],
    ["Status-wise the migration is about seventy percent done, nothing blocking.", "Cool, thanks."],
    ["Sorry I'm late, the train was stuck.", "No worries, we just started."],
    ["Let's give people two more minutes to join.", "Sure."],
    ["Quick update, the dashboards are green, same as yesterday.", "Nice."],
    ["Is it raining where you are too?", "Pouring. Since Monday."],
    ["I'll share my screen, tell me if you can see it.", "Yep, we see the slides."],
    ["Okay, recording is on, everyone fine with that?", "Fine."],
    ["The support queue is normal this week, nothing new.", "Good to hear."],
    ["Happy birthday Marcus, by the way!", "Ha, thank you."],
    ["Can someone drop the doc link in the chat?", "Done, it's in there."],
]

NEVER: list[dict[str, Any]] = [
    {
        "topic": "layoffs",
        "mode": "outright",
        "turns": [
            "The layoffs in support are {canary}, we tell them on the fifteenth.",
            "Okay, keep it quiet until then.",
        ],
        "canaries": ["seventeen people", "eleven people", "twenty two people"],
    },
    {
        "topic": "layoffs",
        "mode": "outright",
        "turns": [
            "On the layoff list, legal wants the {canary} before we send anything.",
            "Fine, we wait for legal.",
        ],
        "canaries": ["severance memo", "WARN filing", "retention letters"],
    },
    {
        "topic": "Project Falcon",
        "mode": "alias",
        "turns": ["Quick Falcon update, the prototype is at {canary}.", "Nice, keep going."],
        "canaries": ["eighty milliseconds", "a thousand users", "version zero nine"],
    },
    {
        "topic": "Project Falcon",
        "mode": "outright",
        "turns": ["Project Falcon needs {canary} more before the demo.", "We can find that."],
        "canaries": ["two engineers", "three weeks", "forty thousand"],
    },
    {
        "topic": "acquisition talks",
        "mode": "alias",
        "turns": ["The Halden deal team asked for {canary}.", "Send it through the lawyers."],
        "canaries": ["the cap table", "our churn cohorts", "the IP schedule"],
    },
    {
        "topic": "acquisition talks",
        "mode": "outright",
        "turns": ["The acquisition talks are at {canary} now.", "That's lower than we hoped."],
        "canaries": ["sixty two million", "fifty eight million", "seventy million"],
    },
    {
        "topic": "layoffs",
        "mode": "ambiguous",
        "turns": [
            "About the headcount reduction we discussed privately, it's {canary} now.",
            "And we tell them the week of the fifteenth?",
            "Yes, keep it in this room.",
        ],
        "canaries": ["nine names", "twelve names", "six names"],
    },
    {
        "topic": "layoffs",
        "mode": "ambiguous",
        "turns": [
            "The cuts we can't talk about yet, finance has them at {canary}.",
            "Okay, not a word outside this call.",
        ],
        "canaries": ["eight percent", "five percent", "ten percent"],
    },
    {
        "topic": "acquisition talks",
        "mode": "ambiguous",
        "turns": [
            "On the thing with the other company, their board wants {canary}.",
            "We go ahead and send it then.",
        ],
        "canaries": [
            "a term sheet by the thirtieth",
            "exclusivity for ninety days",
            "a site visit in December",
        ],
    },
    {
        "topic": "Project Falcon",
        "mode": "ambiguous",
        "turns": [
            "The secret prototype, the bird one, hit {canary} in the lab.",
            "Let's not put that in any notes.",
        ],
        "canaries": ["four hours of battery", "a sub second boot", "ninety five percent accuracy"],
    },
]

OPENERS = [
    "Okay, next thing.",
    "Right, so.",
    "Picking up from before.",
    "On this one, we've gone back and forth.",
    "Let's close this out.",
    "So where did we land?",
]
ACKS = [
    "Yeah, agreed.",
    "Works for me.",
    "Okay, let's do that.",
    "Sure, no objection.",
    "Fine by me.",
    "Makes sense.",
]
THANKS = ["Thanks.", "Great, thank you.", "Perfect.", "Appreciate it."]
WHO_ASKS = ["Who's taking this one?", "Can someone own this?", "Who wants it?"]
FILLERS = ["um,", "uh,", "you know,", "like,", "I mean,", "so,"]
INTERJECTIONS = [
    "-- sorry, go ahead",
    "-- yeah yeah",
    "-- wait, what?",
    "-- mm-hm",
    "-- sorry, you broke up",
]
LEAD_FRAGMENTS = [
    "-- which is why the the numbers looked off.",
    "-- and that's basically the context.",
    "-- so yeah, that's where that came from.",
    "-- anyway, moving on from that.",
]
TAIL_FRAGMENTS = [
    "So the next thing I wanted to",
    "Okay and then on the",
    "Right, and the other",
]
