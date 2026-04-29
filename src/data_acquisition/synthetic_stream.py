import random


LEFT_VOCAB = [
    "equality", "healthcare", "climate", "unions", "diversity",
    "progressive", "welfare", "environment", "minorities", "solidarity",
    "regulation", "public", "reproductive", "justice", "redistribution",
    "green", "inclusion", "workers", "affordable", "subsidies",
    "equity", "feminism", "lgbtq", "abortion", "medicare",
    "medicaid", "renewable", "solar", "emissions", "systemic",
    "racism", "undocumented", "dreamers", "infrastructure", "transit",
    "homelessness", "billionaires", "wealth_tax", "corporate_greed", "safety_net",
    "civil_rights", "voting_rights", "democracy", "diplomacy", "multilateralism",
    "student_debt", "forgiveness", "minimum_wage", "marginalized", "sustainability"
]

RIGHT_VOCAB = [
    "liberty", "market", "border", "taxes", "military",
    "conservative", "traditional", "faith", "deregulation", "private",
    "freedom", "police", "business", "security", "capitalism",
    "growth", "individual", "patriotism", "competition", "family",
    "constitution", "firearms", "pro_life", "unborn", "religious",
    "defense", "veterans", "privatization", "charter_schools", "choice",
    "small_business", "job_creators", "fossil_fuels", "drill", "independence",
    "sovereignty", "enforcement", "wall", "deportation", "aliens",
    "law_and_order", "blue_lives", "originalism", "state_rights", "inflation",
    "socialism", "woke", "spending_cuts", "second_amendment", "heritage"
]

NEUTRAL_VOCAB = [
    "economy", "congress", "president", "law", "policy", "budget", "voters", "election",
    "court", "state", "government", "bill", "senate", "campaign", "debate", "citizens",
    "future", "system", "nation", "power", "bipartisan", "legislation", "representative",
    "governor", "mayor", "municipality", "supreme", "federal", "local", "national",
    "international", "foreign", "domestic", "revenue", "debt", "deficit", "administration",
    "executive", "legislative", "judicial", "candidate", "ballot", "polling", "demographic",
    "committee", "hearing", "amendment", "veto", "council", "agency", "department",
    "official", "spokesperson", "authority", "jurisdiction", "regulation", "statute",
    "media", "press", "news", "speech", "address", "washington", "politics", "interview",
    "statement", "report", "article", "journalist", "broadcast", "network", "coverage",
    "investigation", "source", "editorial", "conference", "briefing", "announcement",
    "publication", "headline", "quote", "document", "testimony", "transcript",
    "people", "community", "society", "world", "global", "family", "children", "school",
    "education", "hospital", "infrastructure", "highway", "bridge", "road", "city",
    "town", "county", "district", "region", "population", "resident", "workforce",
    "industry", "company", "business", "market", "workplace", "institution", "organization",
    "today", "yesterday", "tomorrow", "week", "month", "year", "annual", "history",
    "recent", "current", "event", "schedule", "deadline", "period", "decade", "century",
    "moment", "anniversary", "timeline", "statistics", "data", "average", "total",
    "change", "plan", "proposal", "decision", "action", "result", "impact", "effect",
    "process", "program", "project", "development", "situation", "condition", "issue",
    "problem", "solution", "strategy", "goal", "effort", "challenge", "opportunity",
    "focus", "attention", "support", "opposition", "discussion", "agreement"
]


class SyntheticPoliticalStream:
    def __init__(self, stream_type="basic", seed=42):
        self.rng = random.Random(seed)

        self.left_words = LEFT_VOCAB.copy()
        self.right_words = RIGHT_VOCAB.copy()
        self.neutral_words = NEUTRAL_VOCAB.copy()

        self.stream_type = stream_type
        self.is_drifted = False

        if stream_type == "advanced":
            self.original_left = LEFT_VOCAB.copy()
            self.original_right = RIGHT_VOCAB.copy()

    def generate_sentence(
        self,
        true_class,
        length=10,
        true_ratio=0.50,
        neutral_ratio=0.40,
        opposite_ratio=0.10
    ):
        sentence_words = []

        true_count = int(length * true_ratio)
        opposite_count = int(length * opposite_ratio)
        neutral_count = length - true_count - opposite_count

        if true_class == 0:
            sentence_words.extend(self.rng.choices(self.left_words, k=true_count))
            sentence_words.extend(self.rng.choices(self.right_words, k=opposite_count))
        else:
            sentence_words.extend(self.rng.choices(self.right_words, k=true_count))
            sentence_words.extend(self.rng.choices(self.left_words, k=opposite_count))

        sentence_words.extend(self.rng.choices(self.neutral_words, k=neutral_count))
        self.rng.shuffle(sentence_words)

        return " ".join(sentence_words)

    def trigger_concept_drift(self, drift_ratio, step):
        vocab_size = len(self.left_words)
        num_words_to_swap = int(vocab_size * drift_ratio)

        print(f"\nStep {step}: CONCEPT DRIFT INITIATED")
        self.swap_words(num_words_to_swap)

        self.is_drifted = True

    def swap_words(self, num_words):
        num_words = min(num_words, len(self.left_words))

        if num_words <= 0:
            return

        self.rng.shuffle(self.left_words)
        self.rng.shuffle(self.right_words)

        left_to_right = self.left_words[:num_words]
        right_to_left = self.right_words[:num_words]

        self.left_words = self.left_words[num_words:] + right_to_left
        self.right_words = self.right_words[num_words:] + left_to_right

    def trigger_recurring_concept(self, step):
        if self.stream_type != "advanced":
            raise ValueError("Recurring concepts only available in 'advanced' stream_type")

        print(f"\nStep {step}: RECURRING CONCEPT INITIATED")
        self.left_words = self.original_left.copy()
        self.right_words = self.original_right.copy()

    def get_stream(
        self,
        n_samples=3000,
        drift_points=None,
        drift_ratio=0.6,
        min_len=20,
        max_len=50,
        abrupt_drifts=None,
        gradual_drifts=None,
        recurring_drifts=None
    ):
        if self.stream_type == "basic":
            return self._get_basic_stream(
                n_samples=n_samples,
                drift_points=drift_points,
                drift_ratio=drift_ratio,
                min_len=min_len,
                max_len=max_len
            )

        if self.stream_type == "advanced":
            return self._get_advanced_stream(
                n_samples=n_samples,
                abrupt_drifts=abrupt_drifts,
                gradual_drifts=gradual_drifts,
                recurring_drifts=recurring_drifts,
                min_len=min_len,
                max_len=max_len
            )

        raise ValueError(f"Unknown stream_type: {self.stream_type}")

    def _get_basic_stream(
        self,
        n_samples=3000,
        drift_points=None,
        drift_ratio=0.6,
        min_len=20,
        max_len=50
    ):
        if drift_points is None:
            drift_points = []

        for i in range(n_samples):
            if i in drift_points:
                self.trigger_concept_drift(drift_ratio=drift_ratio, step=i)

            label = self.rng.choice([0, 1])
            doc_length = self.rng.randint(min_len, max_len)
            text = self.generate_sentence(true_class=label, length=doc_length)

            yield text, label

    def _get_advanced_stream(
        self,
        n_samples=3000,
        abrupt_drifts=None,
        gradual_drifts=None,
        recurring_drifts=None,
        min_len=20,
        max_len=50
    ):
        abrupt_drifts = abrupt_drifts or []
        gradual_drifts = gradual_drifts or []
        recurring_drifts = recurring_drifts or []

        gradual_schedule = {}

        for start, end, ratio in gradual_drifts:
            vocab_size = len(self.left_words)
            total_swaps = int(vocab_size * ratio)

            if total_swaps > 0 and end > start:
                interval = max(1, (end - start) // total_swaps)

                for k in range(total_swaps):
                    swap_step = start + (k * interval)

                    if swap_step <= end:
                        gradual_schedule[swap_step] = gradual_schedule.get(swap_step, 0) + 1

            print(f"SCHEDULED: Gradual Drift from {start} to {end} ({total_swaps} words)")

        for i in range(n_samples):
            if i in recurring_drifts:
                self.trigger_recurring_concept(step=i)

            for step, ratio in abrupt_drifts:
                if i == step:
                    swaps = int(len(self.left_words) * ratio)
                    print(f"\nStep {i}: CONCEPT DRIFT INITIATED")
                    self.swap_words(swaps)

            if i in gradual_schedule:
                self.swap_words(gradual_schedule[i])

            label = self.rng.choice([0, 1])
            doc_length = self.rng.randint(min_len, max_len)
            text = self.generate_sentence(true_class=label, length=doc_length)

            yield text, label