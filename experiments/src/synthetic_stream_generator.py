import random
import matplotlib.pyplot as plt
from river import compose, feature_extraction, naive_bayes, metrics, drift, utils


# ==========================================
# 1. POLITICAL DICTIONARIES (US CONTEXT)
# ==========================================

# Words strongly correlated with Left / Democrats (Progressive discourse)
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

# Words strongly correlated with Right / Republicans (Conservative discourse)
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

# Neutral / structural words (Noise used by both sides)
NEUTRAL_VOCAB = [
    # Core Government & Institutions
    "economy", "congress", "president", "law", "policy", "budget", "voters", "election", 
    "court", "state", "government", "bill", "senate", "campaign", "debate", "citizens", 
    "future", "system", "nation", "power", "bipartisan", "legislation", "representative", 
    "governor", "mayor", "municipality", "supreme", "federal", "local", "national", 
    "international", "foreign", "domestic", "revenue", "debt", "deficit", "administration",
    "executive", "legislative", "judicial", "candidate", "ballot", "polling", "demographic", 
    "committee", "hearing", "amendment", "veto", "council", "agency", "department", 
    "official", "spokesperson", "authority", "jurisdiction", "regulation", "statute",

    # Media, Journalism & Communication
    "media", "press", "news", "speech", "address", "washington", "politics", "interview", 
    "statement", "report", "article", "journalist", "broadcast", "network", "coverage", 
    "investigation", "source", "editorial", "conference", "briefing", "announcement", 
    "publication", "headline", "quote", "document", "testimony", "transcript",

    # General Society, Geography & Infrastructure
    "people", "community", "society", "world", "global", "family", "children", "school", 
    "education", "hospital", "infrastructure", "highway", "bridge", "road", "city", 
    "town", "county", "district", "region", "population", "resident", "workforce", 
    "industry", "company", "business", "market", "workplace", "institution", "organization",

    # Time, Metrics & Events
    "today", "yesterday", "tomorrow", "week", "month", "year", "annual", "history", 
    "recent", "current", "event", "schedule", "deadline", "period", "decade", "century", 
    "moment", "anniversary", "timeline", "statistics", "data", "average", "total",

    # Abstract / Action / General Context
    "change", "plan", "proposal", "decision", "action", "result", "impact", "effect", 
    "process", "program", "project", "development", "situation", "condition", "issue", 
    "problem", "solution", "strategy", "goal", "effort", "challenge", "opportunity", 
    "focus", "attention", "support", "opposition", "discussion", "agreement"
]


def create_fresh_model():
    """Returns a fresh TF-IDF + Multinomial Naive Bayes pipeline."""
    return compose.Pipeline(
        ('tfidf', feature_extraction.TFIDF()),
        ('nb', naive_bayes.MultinomialNB())
    )

class SyntheticPoliticalStream:
    """
    Unified generator for synthetic political text streams with concept drift.
    
    Parameters:
        stream_type (str): Type of stream generation
            - 'basic': Simple drift with instant vocabulary swaps
            - 'advanced': Complex drift with abrupt, gradual, and recurring concepts
        train_strategy (str): Model training strategy (only for run_adaptive_experiment)
            - 'all': Train ALL models (active + shadow) on each sample
            - 'active_only': Train ONLY the active model (shadows are frozen)
    """
    
    def __init__(self, stream_type='basic', train_strategy='all'):
        # Store vocabularies in instance variables so we can modify them (simulate drift)
        self.left_words = LEFT_VOCAB.copy()
        self.right_words = RIGHT_VOCAB.copy()
        self.neutral_words = NEUTRAL_VOCAB.copy()
        self.is_drifted = False
        
        # For advanced features
        self.stream_type = stream_type
        self.train_strategy = train_strategy
        
        if stream_type == 'advanced':
            # Store the ORIGINAL state to simulate Recurring Concepts (e.g., new elections)
            self.original_left = LEFT_VOCAB.copy()
            self.original_right = RIGHT_VOCAB.copy()

    def generate_sentence(self, true_class, length=10, true_ratio=0.50, neutral_ratio=0.40, opposite_ratio=0.10):
        """
        Generates a single bag-of-words sentence with realistic noise and cross-contamination.
        true_class: 0 for Left, 1 for Right
        """
        sentence_words = []
        
        # Calculate how many words are assigned to each pool
        true_count = int(length * true_ratio)
        opposite_count = int(length * opposite_ratio)
        # Neutral words fill the rest (safeguard against rounding errors)
        neutral_count = length - true_count - opposite_count
        
        if true_class == 0:  # Left-leaning article
            sentence_words.extend(random.choices(self.left_words, k=true_count))
            sentence_words.extend(random.choices(self.right_words, k=opposite_count))  # Inject right-wing words
        else:  # Right-leaning article
            sentence_words.extend(random.choices(self.right_words, k=true_count))
            sentence_words.extend(random.choices(self.left_words, k=opposite_count))  # Inject left-wing words
            
        # Add neutral noise
        sentence_words.extend(random.choices(self.neutral_words, k=neutral_count))
        
        # Shuffle order to spread words throughout the text
        random.shuffle(sentence_words)
        
        return " ".join(sentence_words)

    def trigger_concept_drift(self, drift_ratio, step):
        """
        SIMULATING DRIFT: 
        Calculates the number of words based on ratio and uses the unified swap_words method.
        """
        vocab_size = len(self.left_words)
        num_words_to_swap = int(vocab_size * drift_ratio)

        percentage_str = int(drift_ratio * 100)
        print(f"\nStep {step}: CONCEPT DRIFT INITIATED")
        
        self.swap_words(num_words_to_swap)
            
        self.is_drifted = True

    def swap_words(self, num_words):
        """Core mathematical logic to physically swap words between buckets."""
        num_words = min(num_words, len(self.left_words))
        if num_words <= 0:
            return

        # Shuffle to pick random words, preventing the same words from swapping back
        random.shuffle(self.left_words)
        random.shuffle(self.right_words)

        left_to_right = self.left_words[:num_words]
        right_to_left = self.right_words[:num_words]

        self.left_words = self.left_words[num_words:] + right_to_left
        self.right_words = self.right_words[num_words:] + left_to_right

    def trigger_recurring_concept(self, step):
        """Instantly restores the vocabularies to their original Day 1 state."""
        if self.stream_type != 'advanced':
            raise ValueError("Recurring concepts only available in 'advanced' stream_type")
        
        print(f"\nStep {step}: RECURRING CONCEPT INITIATED")
        self.left_words = self.original_left.copy()
        self.right_words = self.original_right.copy()

    def get_stream(self, n_samples=3000, drift_points=None, drift_ratio=0.6, min_len=20, max_len=50,
                   abrupt_drifts=None, gradual_drifts=None, recurring_drifts=None):
        """
        Generator function yielding (X, y) where X is text and y is the label.
        
        For 'basic' stream_type:
            - drift_points: list of steps where drift occurs
            - drift_ratio: ratio of words to swap
        
        For 'advanced' stream_type:
            - abrupt_drifts: list of tuples -> [(step, ratio)]
            - gradual_drifts: list of tuples -> [(start_step, end_step, ratio)]
            - recurring_drifts: list of ints -> [step]
        """
        if self.stream_type == 'basic':
            return self._get_basic_stream(n_samples, drift_points, drift_ratio, min_len, max_len)
        elif self.stream_type == 'advanced':
            return self._get_advanced_stream(n_samples, abrupt_drifts, gradual_drifts, recurring_drifts, min_len, max_len)
        else:
            raise ValueError(f"Unknown stream_type: {self.stream_type}")

    def _get_basic_stream(self, n_samples=3000, drift_points=None, drift_ratio=0.6, min_len=20, max_len=50):
        """Basic stream with simple drift points."""
        if drift_points is None:
            drift_points = []
            
        for i in range(n_samples):
            if i in drift_points:
                self.trigger_concept_drift(drift_ratio=drift_ratio, step=i)
                
            label = random.choice([0, 1]) 
            
            # Randomize the length for each specific article
            doc_length = random.randint(min_len, max_len)
            
            # Generate the text with the randomized length
            text = self.generate_sentence(true_class=label, length=doc_length)
            
            yield text, label

    def _get_advanced_stream(self, n_samples=3000, abrupt_drifts=None, gradual_drifts=None, 
                             recurring_drifts=None, min_len=20, max_len=50):
        """Advanced stream with multiple drift types."""
        if abrupt_drifts is None:
            abrupt_drifts = []
        if gradual_drifts is None:
            gradual_drifts = []
        if recurring_drifts is None:
            recurring_drifts = []

        # 1. Pre-compute the Gradual Drift schedule (Step -> Words to Swap)
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
            print(f"SCHEDULED: Gradual Drift from {start} to {end} (Total {total_swaps} words)")
            

        # 2. Main Stream Loop
        for i in range(n_samples):
            # A. Check for Recurring
            if i in recurring_drifts:
                self.trigger_recurring_concept(step=i)

            # B. Check for Abrupt
            for step, ratio in abrupt_drifts:
                if i == step:
                    swaps = int(len(self.left_words) * ratio)
                    print(f"\nStep {i}: CONCEPT DRIFT INITIATED")
                    self.swap_words(swaps)

            # C. Check for Gradual
            if i in gradual_schedule:
                self.swap_words(gradual_schedule[i])
                

            label = random.choice([0, 1]) 
            doc_length = random.randint(min_len, max_len)
            text = self.generate_sentence(true_class=label, length=doc_length)
            
            yield text, label
    
    
    def run_adaptive_experiment(self, stream, reset_on_drift=True):
            """
            Runs the data stream simulation.
            If reset_on_drift=True, it creates a new model upon every drift alert,
            but keeps the old ones running in the background as "Shadow Models" 
            for comparative analysis.
            
            Training strategy is controlled by self.train_strategy:
            - 'all': Train ALL models (active and shadow) on each new sample
            - 'active_only': Train ONLY the active model (shadow models are frozen)
            """
            print(f"Starting the experiment...")
            
            # List to store all models and their performance history.
            # We start with one main active model.
            models_info = [{
                'model': create_fresh_model(),
                'metric': utils.Rolling(metrics.Accuracy(), window_size=100),
                'accuracies': [], 
                'start_step': 0
            }]
            
            detector = drift.ADWIN(delta=0.02)
            
            steps = []
            drifts_detected = []
            
            for i, (text, y_true) in enumerate(stream):
                steps.append(i)
                
                # The index of the active (ruling) model is always the last one added
                active_model_idx = len(models_info) - 1
                
                # 1. Evaluate and train models based on strategy
                for idx, info in enumerate(models_info):
                    y_pred = info['model'].predict_one(text)
                    
                    if y_pred is not None:
                        info['metric'].update(y_true, y_pred)
                        
                        # ADWIN only monitors the errors of the NEWEST (active) model
                        if idx == active_model_idx:
                            error = 0 if y_pred == y_true else 1
                            detector.update(error)
                    
                    # Save the history for the plot
                    info['accuracies'].append(info['metric'].get())
                    
                    # Train based on strategy
                    if self.train_strategy == 'all':
                        # Train all models
                        info['model'].learn_one(text, y_true)
                    elif self.train_strategy == 'active_only':
                        # Train only the active model
                        if idx == active_model_idx:
                            info['model'].learn_one(text, y_true)
                    else:
                        raise ValueError(f"Unknown train_strategy: {self.train_strategy}")
                    
                # 2. React to Concept Drift
                if detector.drift_detected:
                    drifts_detected.append(i)
                    print(f"\nStep {i}: Concept Drift detected on Model {active_model_idx + 1}")
                    
                    if reset_on_drift:
                        if self.train_strategy == 'active_only':
                            print(f"Creating New Model {active_model_idx + 2}. Old model frozen as shadow.")
                        else:
                            print(f"Creating New Model {active_model_idx + 2}. Old model moved to shadow.")
                        
                        # Create a new model and fill its "past" with None values,
                        # so the curves align perfectly on the X-axis of the plot.
                        new_info = {
                            'model': create_fresh_model(),
                            'metric': utils.Rolling(metrics.Accuracy(), window_size=100),
                            'accuracies': [None] * (i + 1), 
                            'start_step': i + 1
                        }
                        models_info.append(new_info)
                        
                        # Reset ADWIN so it doesn't trigger based on the predecessor's errors
                        detector = drift.ADWIN(delta=0.02)

            return steps, models_info, drifts_detected

    