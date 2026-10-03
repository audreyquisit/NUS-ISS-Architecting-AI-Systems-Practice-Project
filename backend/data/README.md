# Hawker centre catalogue

`hawker_centres.json` is based on NEA's **Hawker Centres (GEOJSON)** dataset,
published through data.gov.sg:

- Dataset: <https://data.gov.sg/datasets/d_4a086da0a5553be1d89383cd90d07ecd/view>
- Coverage shown by data.gov.sg: November 2025
- NEA coordinates are WGS84 GeoJSON longitude/latitude pairs, stored here as
  `longitude` and `latitude`.

The catalogue contains the 123 records whose NEA status is `Existing`,
`Existing (new)`, `Existing (replacement)`, or `Interim Centre`. The six
`Under Construction` records are omitted from recommendations. Each entry
retains its NEA name, address, status, cooked-food-stall count, and object ID.
The four pre-existing pilot centre IDs and user-facing names are preserved;
their NEA names are retained in `official_name`/`aliases`.

NEA publishes this data under the data.gov.sg Open Data Licence. Refresh this
file from the official dataset when centre records change; stall counts and
centre status should not be treated as live queue or stall-availability data.

## DBS PayLah! hawker-centre merchant list

`hawker_stall_directory.json` is a list of merchants participating in the DBS PayLah! Saturdays
promotion, not a complete or current directory of every hawker stall. It does
not provide menus, prices, dietary suitability, or opening hours. Do not infer
those fields from a stall name, or use this directory alone to claim that a
stall satisfies a food or budget preference. Check the source PDF when exact
spelling or unit details matter.

## Wak-Wak Hawker stall and menu catalogue

`hawker_stall.json` contains the scraped stall directory and menu listings from
<https://wak-wak-hawker.com/en/stalls/>. The source includes stall names,
locations, categories, menu names, and source price strings. Its publication or
last-updated date is not recorded here, so treat menu availability and prices as
reference data rather than live listings.

Each stall has a stable local `stall_id`. `hawker_centre_id` is populated only
when the stall's six-digit postcode exactly matches one unique NEA centre
postcode; records without that match remain in the source file but are not used
as verified recommendations. Menu rows have stable `menu_item_id` values.

Price normalization preserves the original string in both `price` and
`price_raw`, and adds `price_sgd`, `price_sgd_min`, `price_sgd_max`,
`price_sgd_options`, `price_option_labels`, `price_model`, `price_status`, and
`price_basis`. Only explicit numeric SGD amounts are parsed. Two- or three-price
slash-separated menu rows without variant labels are represented as portion
tiers; other multi-price rows remain distinct listed options. Ingredient-choice rows that explicitly say the final price depends on
the selected ingredients are marked `ingredient_based`, with no invented
amount. Other unknown or non-numeric entries remain unavailable. Some source
prices do not state whether they are per serving, so `price_basis` may be
`unknown`.
Re-run `python scripts/normalize_hawker_stalls.py` from `backend/` after
replacing or refreshing the source file.

The source does not provide structured ingredients, allergens, dietary
certification, opening status, or reliable serving-size data. A stall or dish
name/category must not be treated as evidence that an item is halal, vegetarian,
vegan, or allergy-safe. Budget filtering currently compares the user's budget
with each menu item's listed starting price, not the total of a meal.

## Synthetic demo coverage

`hawker_stall_mock.json` adds clearly labelled synthetic examples for NEA
centres that have no stalls in the Wak-Wak catalogue. Each such centre receives
demo chicken rice, wanton mee, and hokkien mee entries with illustrative prices.
These are fictional stalls and menus, not observed businesses or source-backed
prices. The API and recommendation flow carry `is_mock`/`data_origin` metadata
so recommendations can label them as demo data. Regenerate deterministically
with `python scripts/generate_mock_hawker_stalls.py` from `backend/`.
