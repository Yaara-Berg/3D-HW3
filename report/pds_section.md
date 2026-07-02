# Task 2: Posterior Distillation Sampling (PDS)

**Method:** PDS with `guidance_scale = 7.5` (fixed per assignment).  
**Evaluation:** CLIP ViT-B/32 (`eval.py`), scored against each **edit prompt**.  
**Overall CLIP score:** **0.326**

## Summary

| # | Source prompt | Edit prompt | CLIP |
|---|---------------|-------------|------|
| 1 | A red bus driving on a desert road | A yellow school bus driving on a desert road | 0.353 |
| 2 | a boat in a river | a golden boat in a river | 0.295 |
| 3 | A cabin surrounded by forests | A cabin surrounded by flowers | 0.320 |
| 4 | A church beside a lake | A church beside a lake at night | 0.355 |
| 5 | A villa close to the pool | A villa covered with ivy close to the pool | 0.275 |
| 6 | A castle next to a river | A rainbow castle next to a river | 0.351 |
| 7 | A burger on the table | A burger with lettuce on the table | 0.291 |
| 8 | A dog sitting on grass | A golden retriever sitting on grass | 0.324 |
| 9 | a cat sitting on a table | a wooden cat statue sitting on a table | 0.373 |
| 10 | A car on the road | A yellow sportscar on the road | 0.326 |

---

## Results

### 1. A red bus driving on a desert road → A yellow school bus driving on a desert road

**CLIP:** 0.353

| Source | Edited |
|--------|--------|
| ![Source](../data/imgs/A_red_bus_driving_on_a_desert_road.png) | ![Edited](../outputs/pds/A_yellow_school_bus_driving_on_a_desert_road.png) |

### 2. a boat in a river → a golden boat in a river

**CLIP:** 0.295

| Source | Edited |
|--------|--------|
| ![Source](../data/imgs/a_boat_in_a_river.png) | ![Edited](../outputs/pds/a_golden_boat_in_a_river.png) |

### 3. A cabin surrounded by forests → A cabin surrounded by flowers

**CLIP:** 0.320

| Source | Edited |
|--------|--------|
| ![Source](../data/imgs/A_cabin_surrounded_by_forests.png) | ![Edited](../outputs/pds/A_cabin_surrounded_by_flowers.png) |

### 4. A church beside a lake → A church beside a lake at night

**CLIP:** 0.355

| Source | Edited |
|--------|--------|
| ![Source](../data/imgs/A_church_beside_a_lake.png) | ![Edited](../outputs/pds/A_church_beside_a_lake_at_night.png) |

### 5. A villa close to the pool → A villa covered with ivy close to the pool

**CLIP:** 0.275

| Source | Edited |
|--------|--------|
| ![Source](../data/imgs/A_villa_close_to_the_pool.png) | ![Edited](../outputs/pds/A_villa_covered_with_ivy_close_to_the_pool.png) |

### 6. A castle next to a river → A rainbow castle next to a river

**CLIP:** 0.351

| Source | Edited |
|--------|--------|
| ![Source](../data/imgs/A_castle_next_to_a_river.png) | ![Edited](../outputs/pds/A_rainbow_castle_next_to_a_river.png) |

### 7. A burger on the table → A burger with lettuce on the table

**CLIP:** 0.291

| Source | Edited |
|--------|--------|
| ![Source](../data/imgs/A_burger_on_the_table.png) | ![Edited](../outputs/pds/A_burger_with_lettuce_on_the_table.png) |

### 8. A dog sitting on grass → A golden retriever sitting on grass

**CLIP:** 0.324

| Source | Edited |
|--------|--------|
| ![Source](../data/imgs/A_dog_sitting_on_grass.png) | ![Edited](../outputs/pds/A_golden_retriever_sitting_on_grass.png) |

### 9. a cat sitting on a table → a wooden cat statue sitting on a table

**CLIP:** 0.373

| Source | Edited |
|--------|--------|
| ![Source](../data/imgs/a_cat_sitting_on_a_table.png) | ![Edited](../outputs/pds/a_wooden_cat_statue_sitting_on_a_table.png) |

### 10. A car on the road → A yellow sportscar on the road

**CLIP:** 0.326

| Source | Edited |
|--------|--------|
| ![Source](../data/imgs/A_car_on_the_road.png) | ![Edited](../outputs/pds/A_yellow_sportscar_on_the_road.png) |
