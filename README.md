# Roommate Compatibility Analysis

_Modeling Roommate Compatibility Through Psychological and Behavioral Data_

This project explores how individuals with different psychological, behavioral, and lifestyle traits can be grouped into compatible roommate matches. It addresses a common question for anyone living in a shared space, such as dormitories: _"Who would be a better roommate for me?"_

To achieve this, a large-scale dataset of **100,000 records and 50 features** was analyzed using various clustering techniques, including **K-Means, Hierarchical Clustering, DBSCAN, and Spectral Clustering**. In addition, a **compatibility scoring system** was developed, combining clustering results with cosine similarity to assign each pair of individuals a numerical compatibility score.


## Motivation

Choosing the right roommate(s) can have a big impact on a person's mental health, productivity and overall quality of life. Living with someone who isn't compatible can easily lead to stress, arguments and an uncomfortable home environment — especially in shared spaces like dormitories or student housing.

Traditional roommate matching methods usually focus on only a few simple preferences such as cleanliness or sleep schedules and ignore deeper psychological or behavioral traits. This project aims to close that gap by using data-driven clustering and similarity analysis to discover meaningful compatibility patterns between people.

By analyzing psychological, behavioral and lifestyle features, this project shows how machine learning can be used to create more balanced and comfortable living environments through clear and objective compatibility analysis.

## Dataset
The dataset used in this project was extracted from a [public GitHub repository](https://github.com/Pradyumna-yes/Roommate-Pairing-ML-Model). It contains **100,000 rows and 50 features** representing **psychological, behavioral and lifestyle traits**, with all feature values ranging from **0 to 5**. 

The dataset was collected through self-reported responses to a comprehensive questionnaire. The questions were designed based on validated psychological and behavioral criteria, including the [IPIP Big-Five Factor Markers](https://openpsychometrics.org/tests/IPIP-BFFM/) and related resources from the [International Personality Item Pool (IPIP)](https://ipip.ori.org/new_ipip-50-item-scale.htm). This ensured that the data captured reliable psychological and behavioral dimensions, making it suitable for analytical and clustering approaches.

Some algorithms were applied directly to the original dataset, while others used the **`compressed_features`** version, which groups related features together. The mean value of each feature group represents a higher-level characteristic. In total, **seven feature groups** were created.

1. **Lifestyle**
    - Sleeping Patterns 
    - Eating Habits
    - Exercise Routines
    - Cleanliness and Hygiene
    - Smoking and Drinking Habits
    - Attitudes Toward Money and Finances
    - Preference for Quiet vs. Lively Environment
    - Views on Personal and Shared Spaces

2. **Interests & Activities**
    - Music Preferences
    - Movie/TV Show Genres
    - Reading and Literature
    - Sports and Outdoor Activities
    - Cooking and Culinary Interests
    - Travel and Exploration Interests
    - Artistic Interests 
    - Creativity 

3. **Communication & Interaction**
    - Listening Skills
    - Conflict Resolution Abilities
    - Openness to Feedback
    - Expressiveness
    - Empathy in Communication
    - Frequency of Hosting Guests
    - Participation in Community Events
    - Networking and Socializing Habits

4. **Responsibility & Organization**
    - Work Ethic
    - Organizational Skills
    - Reliability
    - Punctuality
    - Attention to Detail

5. **Social Engagement & Behavior**
    - Talkativeness
    - Enthusiasm for Social Gatherings
    - Energy Levels
    - Sociability
    - Assertiveness
    - Cooperation
    - Trust in Others
    - Altruism
    - Modesty

6. **Emotional Regulation & Stability**
    - Compassion
    - Anger Management
    - Stress Management
    - Emotional Stability
    - Anxiety Levels
    - Mood Swings

7. **Openness & Core Beliefs**
    - Intellectual Curiosity
    - Preference for Novelty
    - Openness to Diverse Cultures
    - Political Views
    - Religious Beliefs
    - Cultural Values