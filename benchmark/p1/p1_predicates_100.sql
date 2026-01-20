-- Q7 base true_cardinality: 3323
-- Generated with seed=20260115

-- q7_p001
-- predicates: phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND c.creationdate BETWEEN 1340327339000 AND 1356461058629 AND chp.explicitlydeleted = false
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND c.creationdate BETWEEN 1340327339000 AND 1356461058629 AND chp.explicitlydeleted = false; -- || 1468

-- q7_p002
-- predicates: chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND t.name = 'Augustine_of_Hippo' AND cht.creationdate BETWEEN 1338765911761 AND 1356307644022
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND t.name = 'Augustine_of_Hippo' AND cht.creationdate BETWEEN 1338765911761 AND 1356307644022; -- || 103

-- q7_p003
-- predicates: cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND p.language = 'pt;en'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND p.language = 'pt;en'; -- || 99

-- q7_p004
-- predicates: chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND t.name = 'Plato'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND t.name = 'Plato'; -- || 23

-- q7_p005
-- predicates: cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND p.language = 'pt;en' AND phi.creationdate BETWEEN 1285156166915 AND 1348051459240
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND p.language = 'pt;en' AND phi.creationdate BETWEEN 1285156166915 AND 1348051459240; -- || 63

-- q7_p006
-- predicates: cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND p.gender = 'male'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND p.gender = 'male'; -- || 1180

-- q7_p007
-- predicates: phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND t.name = 'George_W._Bush' AND p.language = 'en'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND t.name = 'George_W._Bush' AND p.language = 'en'; -- || 1

-- q7_p008
-- predicates: cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND c.creationdate >= 1350093427787 AND chp.creationdate BETWEEN 1340327339000 AND 1356461058629
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND c.creationdate >= 1350093427787 AND chp.creationdate BETWEEN 1340327339000 AND 1356461058629; -- || 1234

-- q7_p009
-- predicates: chp.explicitlydeleted = false AND c.explicitlydeleted = false AND phi.creationdate BETWEEN 1285156166915 AND 1348051459240
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE chp.explicitlydeleted = false AND c.explicitlydeleted = false AND phi.creationdate BETWEEN 1285156166915 AND 1348051459240; -- || 2150

-- q7_p010
-- predicates: cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND c.creationdate BETWEEN 1340327339000 AND 1356461058629 AND chp.explicitlydeleted = false
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND c.creationdate BETWEEN 1340327339000 AND 1356461058629 AND chp.explicitlydeleted = false; -- || 2078

-- q7_p011
-- predicates: cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND p.gender = 'female' AND c.length >= 79
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND p.gender = 'female' AND c.length >= 79; -- || 796

-- q7_p012
-- predicates: cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND p.language = 'en'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND p.language = 'en'; -- || 211

-- q7_p013
-- predicates: chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND c.explicitlydeleted = false AND cht.creationdate BETWEEN 1338765911761 AND 1356307644022
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND c.explicitlydeleted = false AND cht.creationdate BETWEEN 1338765911761 AND 1356307644022; -- || 2078

-- q7_p014
-- predicates: cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND c.creationdate BETWEEN 1340327339000 AND 1356461058629 AND chp.creationdate BETWEEN 1340327339000 AND 1356461058629
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND c.creationdate BETWEEN 1340327339000 AND 1356461058629 AND chp.creationdate BETWEEN 1340327339000 AND 1356461058629; -- || 2109

-- q7_p015
-- predicates: cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND p.language = 'en' AND chp.creationdate BETWEEN 1340327339000 AND 1356461058629
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND p.language = 'en' AND chp.creationdate BETWEEN 1340327339000 AND 1356461058629; -- || 202

-- q7_p016
-- predicates: cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND t.name = 'George_W._Bush'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND t.name = 'George_W._Bush'; -- || 19

-- q7_p017
-- predicates: p.language = 'zh;en' AND t.name = 'Jesus'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE p.language = 'zh;en' AND t.name = 'Jesus'; -- || 6

-- q7_p018
-- predicates: chp.explicitlydeleted = false AND t.name = 'Augustine_of_Hippo' AND cht.creationdate BETWEEN 1338765911761 AND 1356307644022
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE chp.explicitlydeleted = false AND t.name = 'Augustine_of_Hippo' AND cht.creationdate BETWEEN 1338765911761 AND 1356307644022; -- || 106

-- q7_p019
-- predicates: chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND c.length >= 79
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND c.length >= 79; -- || 1636

-- q7_p020
-- predicates: cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND t.name = 'William_Shakespeare'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND t.name = 'William_Shakespeare'; -- || 18

-- q7_p021
-- predicates: chp.explicitlydeleted = false AND t.name = 'Jesus'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE chp.explicitlydeleted = false AND t.name = 'Jesus'; -- || 21

-- q7_p022
-- predicates: chp.explicitlydeleted = false AND c.length >= 79
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE chp.explicitlydeleted = false AND c.length >= 79; -- || 2513

-- q7_p023
-- predicates: t.name = 'Elizabeth_II' AND c.explicitlydeleted = false AND cht.creationdate BETWEEN 1338765911761 AND 1356307644022
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE t.name = 'Elizabeth_II' AND c.explicitlydeleted = false AND cht.creationdate BETWEEN 1338765911761 AND 1356307644022; -- || 13

-- q7_p024
-- predicates: cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND c.length >= 79 AND t.name = 'Plato'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND c.length >= 79 AND t.name = 'Plato'; -- || 17

-- q7_p025
-- predicates: cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND t.name = 'Plato'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND t.name = 'Plato'; -- || 24

-- q7_p026
-- predicates: cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND p.language = 'es;en'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND p.language = 'es;en'; -- || 114

-- q7_p027
-- predicates: chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p.language = 'zh;en' AND phi.creationdate BETWEEN 1285156166915 AND 1348051459240
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p.language = 'zh;en' AND phi.creationdate BETWEEN 1285156166915 AND 1348051459240; -- || 304

-- q7_p028
-- predicates: chp.explicitlydeleted = false AND p.gender = 'male'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE chp.explicitlydeleted = false AND p.gender = 'male'; -- || 1714

-- q7_p029
-- predicates: chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p.language = 'es;en'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p.language = 'es;en'; -- || 108

-- q7_p030
-- predicates: cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND t.name = 'Plato' AND phi.creationdate BETWEEN 1285156166915 AND 1348051459240
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND t.name = 'Plato' AND phi.creationdate BETWEEN 1285156166915 AND 1348051459240; -- || 18

-- q7_p031
-- predicates: phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND p.language = 'es;en'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND p.language = 'es;en'; -- || 101

-- q7_p032
-- predicates: phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND t.name = 'Plato'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND t.name = 'Plato'; -- || 24

-- q7_p033
-- predicates: chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND c.length BETWEEN 5 AND 79
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND c.length BETWEEN 5 AND 79; -- || 610

-- q7_p034
-- predicates: chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND c.creationdate BETWEEN 1340327339000 AND 1356461058629
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND c.creationdate BETWEEN 1340327339000 AND 1356461058629; -- || 2142

-- q7_p035
-- predicates: cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND p.language = 'zh;en'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND p.language = 'zh;en'; -- || 399

-- q7_p036
-- predicates: cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND t.name = 'Jesus' AND p.gender = 'female'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND t.name = 'Jesus' AND p.gender = 'female'; -- || 8

-- q7_p037
-- predicates: phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND t.name = 'Jesus'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND t.name = 'Jesus'; -- || 16

-- q7_p038
-- predicates: phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND t.name = 'William_Shakespeare'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND t.name = 'William_Shakespeare'; -- || 14

-- q7_p039
-- predicates: chp.explicitlydeleted = false AND t.name = 'Elizabeth_II'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE chp.explicitlydeleted = false AND t.name = 'Elizabeth_II'; -- || 22

-- q7_p040
-- predicates: phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND p.language = 'es;en' AND chp.creationdate BETWEEN 1340327339000 AND 1356461058629
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND p.language = 'es;en' AND chp.creationdate BETWEEN 1340327339000 AND 1356461058629; -- || 74

-- q7_p041
-- predicates: phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND p.language = 'zh;en' AND chp.creationdate BETWEEN 1340327339000 AND 1356461058629
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND p.language = 'zh;en' AND chp.creationdate BETWEEN 1340327339000 AND 1356461058629; -- || 304

-- q7_p042
-- predicates: chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND t.name = 'Aristotle' AND c.length BETWEEN 5 AND 79
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND t.name = 'Aristotle' AND c.length BETWEEN 5 AND 79; -- || 2

-- q7_p043
-- predicates: chp.explicitlydeleted = false AND c.explicitlydeleted = false AND cht.creationdate BETWEEN 1338765911761 AND 1356307644022
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE chp.explicitlydeleted = false AND c.explicitlydeleted = false AND cht.creationdate BETWEEN 1338765911761 AND 1356307644022; -- || 2169

-- q7_p044
-- predicates: cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND t.name = 'Elizabeth_II'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND t.name = 'Elizabeth_II'; -- || 13

-- q7_p045
-- predicates: chp.explicitlydeleted = false AND p.language = 'pt;en'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE chp.explicitlydeleted = false AND p.language = 'pt;en'; -- || 143

-- q7_p046
-- predicates: chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND t.name = 'Napoleon'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND t.name = 'Napoleon'; -- || 18

-- q7_p047
-- predicates: cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND p.gender = 'female'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND p.gender = 'female'; -- || 1023

-- q7_p048
-- predicates: phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND p.gender = 'male'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND p.gender = 'male'; -- || 1207

-- q7_p049
-- predicates: c.length >= 79 AND p.gender = 'male'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE c.length >= 79 AND p.gender = 'male'; -- || 1314

-- q7_p050
-- predicates: phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND p.language = 'pt;en' AND c.length >= 79
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND p.language = 'pt;en' AND c.length >= 79; -- || 63

-- q7_p051
-- predicates: chp.explicitlydeleted = false AND t.name = 'Elizabeth_II' AND phi.creationdate BETWEEN 1285156166915 AND 1348051459240
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE chp.explicitlydeleted = false AND t.name = 'Elizabeth_II' AND phi.creationdate BETWEEN 1285156166915 AND 1348051459240; -- || 17

-- q7_p052
-- predicates: chp.explicitlydeleted = false AND p.language = 'zh;en'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE chp.explicitlydeleted = false AND p.language = 'zh;en'; -- || 579

-- q7_p053
-- predicates: c.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p.gender = 'female' AND cht.creationdate BETWEEN 1338765911761 AND 1356307644022
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE c.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p.gender = 'female' AND cht.creationdate BETWEEN 1338765911761 AND 1356307644022; -- || 982

-- q7_p054
-- predicates: chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND t.name = 'Jesus'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND t.name = 'Jesus'; -- || 14

-- q7_p055
-- predicates: t.name = 'Jesus' AND c.length BETWEEN 5 AND 79 AND cht.creationdate BETWEEN 1338765911761 AND 1356307644022
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE t.name = 'Jesus' AND c.length BETWEEN 5 AND 79 AND cht.creationdate BETWEEN 1338765911761 AND 1356307644022; -- || 3

-- q7_p056
-- predicates: phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND p.gender = 'female'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND p.gender = 'female'; -- || 976

-- q7_p057
-- predicates: cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND t.name = 'John_F._Kennedy' AND phi.creationdate BETWEEN 1285156166915 AND 1348051459240
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND t.name = 'John_F._Kennedy' AND phi.creationdate BETWEEN 1285156166915 AND 1348051459240; -- || 5

-- q7_p058
-- predicates: chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND t.name = 'Bill_Clinton'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND t.name = 'Bill_Clinton'; -- || 9

-- q7_p059
-- predicates: phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND p.gender = 'female' AND c.creationdate BETWEEN 1340327339000 AND 1356461058629
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND p.gender = 'female' AND c.creationdate BETWEEN 1340327339000 AND 1356461058629; -- || 662

-- q7_p060
-- predicates: phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND t.name = 'William_Shakespeare' AND chp.explicitlydeleted = false
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND t.name = 'William_Shakespeare' AND chp.explicitlydeleted = false; -- || 14

-- q7_p061
-- predicates: phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND c.explicitlydeleted = false
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND c.explicitlydeleted = false; -- || 2150

-- q7_p062
-- predicates: cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND c.creationdate >= 1350093427787
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND c.creationdate >= 1350093427787; -- || 1234

-- q7_p063
-- predicates: chp.explicitlydeleted = false AND c.creationdate BETWEEN 1340327339000 AND 1356461058629
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE chp.explicitlydeleted = false AND c.creationdate BETWEEN 1340327339000 AND 1356461058629; -- || 2110

-- q7_p064
-- predicates: cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND c.length >= 79
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND c.length >= 79; -- || 1673

-- q7_p065
-- predicates: c.explicitlydeleted = false AND p.language = 'en'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE c.explicitlydeleted = false AND p.language = 'en'; -- || 319

-- q7_p066
-- predicates: phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND t.name = 'Napoleon' AND cht.creationdate BETWEEN 1338765911761 AND 1356307644022
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND t.name = 'Napoleon' AND cht.creationdate BETWEEN 1338765911761 AND 1356307644022; -- || 11

-- q7_p067
-- predicates: chp.explicitlydeleted = false AND t.name = 'William_Shakespeare'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE chp.explicitlydeleted = false AND t.name = 'William_Shakespeare'; -- || 26

-- q7_p068
-- predicates: p.gender = 'male' AND c.explicitlydeleted = false
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE p.gender = 'male' AND c.explicitlydeleted = false; -- || 1714

-- q7_p069
-- predicates: cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND c.creationdate >= 1350093427787 AND chp.explicitlydeleted = false
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND c.creationdate >= 1350093427787 AND chp.explicitlydeleted = false; -- || 1215

-- q7_p070
-- predicates: cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND t.name = 'John_F._Kennedy' AND chp.explicitlydeleted = false
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND t.name = 'John_F._Kennedy' AND chp.explicitlydeleted = false; -- || 13

-- q7_p071
-- predicates: phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND c.length >= 79 AND chp.creationdate BETWEEN 1340327339000 AND 1356461058629
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND c.length >= 79 AND chp.creationdate BETWEEN 1340327339000 AND 1356461058629; -- || 1140

-- q7_p072
-- predicates: chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND t.name = 'Augustine_of_Hippo'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND t.name = 'Augustine_of_Hippo'; -- || 105

-- q7_p073
-- predicates: phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND t.name = 'Elizabeth_II'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND t.name = 'Elizabeth_II'; -- || 17

-- q7_p074
-- predicates: cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND c.explicitlydeleted = false
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND c.explicitlydeleted = false; -- || 2169

-- q7_p075
-- predicates: phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND t.name = 'George_W._Bush'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND t.name = 'George_W._Bush'; -- || 18

-- q7_p076
-- predicates: cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND c.creationdate BETWEEN 1340327339000 AND 1356461058629
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND c.creationdate BETWEEN 1340327339000 AND 1356461058629; -- || 2109

-- q7_p077
-- predicates: phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND p.language = 'zh;en'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND p.language = 'zh;en'; -- || 454

-- q7_p078
-- predicates: cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND c.length >= 79 AND p.gender = 'female'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND c.length >= 79 AND p.gender = 'female'; -- || 796

-- q7_p079
-- predicates: phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND c.creationdate >= 1350093427787
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND c.creationdate >= 1350093427787; -- || 974

-- q7_p080
-- predicates: chp.explicitlydeleted = false AND c.length >= 79 AND p.language = 'es;en'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE chp.explicitlydeleted = false AND c.length >= 79 AND p.language = 'es;en'; -- || 135

-- q7_p081
-- predicates: chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p.gender = 'male' AND cht.creationdate BETWEEN 1338765911761 AND 1356307644022
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p.gender = 'male' AND cht.creationdate BETWEEN 1338765911761 AND 1356307644022; -- || 1127

-- q7_p082
-- predicates: chp.explicitlydeleted = false AND t.name = 'George_W._Bush'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE chp.explicitlydeleted = false AND t.name = 'George_W._Bush'; -- || 28

-- q7_p083
-- predicates: cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND t.name = 'Augustine_of_Hippo' AND chp.explicitlydeleted = false
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND t.name = 'Augustine_of_Hippo' AND chp.explicitlydeleted = false; -- || 106

-- q7_p084
-- predicates: chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND c.creationdate >= 1350093427787
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND c.creationdate >= 1350093427787; -- || 1267

-- q7_p085
-- predicates: chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p.language = 'zh;en'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p.language = 'zh;en'; -- || 378

-- q7_p086
-- predicates: chp.explicitlydeleted = false AND p.gender = 'female' AND cht.creationdate BETWEEN 1338765911761 AND 1356307644022
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE chp.explicitlydeleted = false AND p.gender = 'female' AND cht.creationdate BETWEEN 1338765911761 AND 1356307644022; -- || 1011

-- q7_p087
-- predicates: chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND t.name = 'George_W._Bush' AND cht.creationdate BETWEEN 1338765911761 AND 1356307644022
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND t.name = 'George_W._Bush' AND cht.creationdate BETWEEN 1338765911761 AND 1356307644022; -- || 19

-- q7_p088
-- predicates: phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND t.name = 'George_W._Bush' AND c.explicitlydeleted = false
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND t.name = 'George_W._Bush' AND c.explicitlydeleted = false; -- || 18

-- q7_p089
-- predicates: chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND c.explicitlydeleted = false
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND c.explicitlydeleted = false; -- || 2110

-- q7_p090
-- predicates: chp.explicitlydeleted = false AND t.name = 'Napoleon'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE chp.explicitlydeleted = false AND t.name = 'Napoleon'; -- || 24

-- q7_p091
-- predicates: chp.explicitlydeleted = false AND p.gender = 'female' AND c.length >= 79
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE chp.explicitlydeleted = false AND p.gender = 'female' AND c.length >= 79; -- || 1220

-- q7_p092
-- predicates: chp.explicitlydeleted = false AND t.name = 'Augustine_of_Hippo'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE chp.explicitlydeleted = false AND t.name = 'Augustine_of_Hippo'; -- || 187

-- q7_p093
-- predicates: chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND c.explicitlydeleted = false AND phi.creationdate BETWEEN 1285156166915 AND 1348051459240
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND c.explicitlydeleted = false AND phi.creationdate BETWEEN 1285156166915 AND 1348051459240; -- || 1468

-- q7_p094
-- predicates: cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND p.gender = 'male' AND chp.creationdate BETWEEN 1340327339000 AND 1356461058629
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND p.gender = 'male' AND chp.creationdate BETWEEN 1340327339000 AND 1356461058629; -- || 1127

-- q7_p095
-- predicates: cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND t.name = 'Jesus'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND t.name = 'Jesus'; -- || 15

-- q7_p096
-- predicates: chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND t.name = 'John_F._Kennedy'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND t.name = 'John_F._Kennedy'; -- || 13

-- q7_p097
-- predicates: phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND c.length >= 79 AND t.name = 'Augustine_of_Hippo'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND c.length >= 79 AND t.name = 'Augustine_of_Hippo'; -- || 101

-- q7_p098
-- predicates: cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND t.name = 'William_Shakespeare' AND chp.explicitlydeleted = false
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE cht.creationdate BETWEEN 1338765911761 AND 1356307644022 AND t.name = 'William_Shakespeare' AND chp.explicitlydeleted = false; -- || 18

-- q7_p099
-- predicates: phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND p.language = 'en'
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND p.language = 'en'; -- || 247

-- q7_p100
-- predicates: phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND c.creationdate BETWEEN 1340327339000 AND 1356461058629
SELECT
  -- 点表
  c.*,
  p.*,
  t.*,

  -- 边表
  cht.*,
  chp.*,
  phi.*
FROM comment AS c
JOIN comment_hastag_tag AS cht
  ON cht.commentid = c.id
JOIN tag AS t
  ON t.id = cht.tagid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p
  ON p.id = chp.personid
JOIN person_hasinterest_tag AS phi
  ON phi.personid = p.id
 AND phi.tagid = t.id
WHERE phi.creationdate BETWEEN 1285156166915 AND 1348051459240 AND c.creationdate BETWEEN 1340327339000 AND 1356461058629; -- || 1493

