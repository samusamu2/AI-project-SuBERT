# 3-glyphs_transliteration

Since there's a deterministic mapping from glyphs to transliterated words, once completed our vision task we could, in principle, transliterate the obtained glyphs and apply our translation models, contained in the next folder.

Here you find the JSON files containing the mapping between glyph names and glyph unicode characters, and the mapping between morphemes (the actual transliteration we are interested in) and glyph names.

Just reversing these mapping we can create a full pipeline from glyphs recognition via visual models to glyphs translation via Transformer models.