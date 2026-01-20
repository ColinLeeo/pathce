-- Q8 base true_cardinality: 2183
-- Generated with seed=20260115

-- q8_p001
-- predicates: plc2.explicitlydeleted = false AND p1.language = 'zh;en' AND chp.explicitlydeleted = false
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc2.explicitlydeleted = false AND p1.language = 'zh;en' AND chp.explicitlydeleted = false; -- || 268

-- q8_p002
-- predicates: plc_like.creationdate >= 1351344255062 AND ci.name = 'Rạch_Giá' AND plc1.explicitlydeleted = false
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc_like.creationdate >= 1351344255062 AND ci.name = 'Rạch_Giá' AND plc1.explicitlydeleted = false; -- || 2

-- q8_p003
-- predicates: plc2.explicitlydeleted = false AND c.creationdate >= 1350093427787 AND p2.language = 'en'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc2.explicitlydeleted = false AND c.creationdate >= 1350093427787 AND p2.language = 'en'; -- || 101

-- q8_p004
-- predicates: plc_like.explicitlydeleted = false AND p2.language = 'pt;en' AND plc2.creationdate BETWEEN 1285358988937 AND 1347940587243
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc_like.explicitlydeleted = false AND p2.language = 'pt;en' AND plc2.creationdate BETWEEN 1285358988937 AND 1347940587243; -- || 41

-- q8_p005
-- predicates: plc_like.creationdate >= 1351344255062 AND p2.language = 'en' AND plc1.creationdate BETWEEN 1285358988937 AND 1347940587243
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc_like.creationdate >= 1351344255062 AND p2.language = 'en' AND plc1.creationdate BETWEEN 1285358988937 AND 1347940587243; -- || 62

-- q8_p006
-- predicates: chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p2.gender = 'female' AND p1.language = 'zh;en'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p2.gender = 'female' AND p1.language = 'zh;en'; -- || 95

-- q8_p007
-- predicates: chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND c.creationdate BETWEEN 1340327339000 AND 1356461058629 AND plc_like.explicitlydeleted = false
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND c.creationdate BETWEEN 1340327339000 AND 1356461058629 AND plc_like.explicitlydeleted = false; -- || 1388

-- q8_p008
-- predicates: plc_like.explicitlydeleted = false AND c.length BETWEEN 5 AND 79 AND plc1.explicitlydeleted = false
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc_like.explicitlydeleted = false AND c.length BETWEEN 5 AND 79 AND plc1.explicitlydeleted = false; -- || 588

-- q8_p009
-- predicates: plc1.explicitlydeleted = false AND c.length BETWEEN 5 AND 79 AND plc_like.creationdate >= 1351344255062
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc1.explicitlydeleted = false AND c.length BETWEEN 5 AND 79 AND plc_like.creationdate >= 1351344255062; -- || 244

-- q8_p010
-- predicates: plc1.explicitlydeleted = false AND p2.language = 'en'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc1.explicitlydeleted = false AND p2.language = 'en'; -- || 188

-- q8_p011
-- predicates: plc2.creationdate BETWEEN 1285358988937 AND 1347940587243 AND p2.language = 'en'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc2.creationdate BETWEEN 1285358988937 AND 1347940587243 AND p2.language = 'en'; -- || 131

-- q8_p012
-- predicates: chp.explicitlydeleted = false AND c.length >= 79 AND plc_like.creationdate >= 1351344255062
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE chp.explicitlydeleted = false AND c.length >= 79 AND plc_like.creationdate >= 1351344255062; -- || 767

-- q8_p013
-- predicates: plc_like.explicitlydeleted = false AND c.length >= 79 AND p2.language = 'pt;en'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc_like.explicitlydeleted = false AND c.length >= 79 AND p2.language = 'pt;en'; -- || 76

-- q8_p014
-- predicates: plc1.creationdate BETWEEN 1285358988937 AND 1347940587243 AND c.length >= 79 AND p1.gender = 'male'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc1.creationdate BETWEEN 1285358988937 AND 1347940587243 AND c.length >= 79 AND p1.gender = 'male'; -- || 542

-- q8_p015
-- predicates: chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND c.creationdate >= 1350093427787
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND c.creationdate >= 1350093427787; -- || 871

-- q8_p016
-- predicates: plc1.explicitlydeleted = false AND p1.gender = 'female' AND p2.gender = 'female'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc1.explicitlydeleted = false AND p1.gender = 'female' AND p2.gender = 'female'; -- || 1112

-- q8_p017
-- predicates: plc_like.explicitlydeleted = false AND p2.language = 'en'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc_like.explicitlydeleted = false AND p2.language = 'en'; -- || 184

-- q8_p018
-- predicates: chp.explicitlydeleted = false AND ci.name = 'Rạch_Giá' AND plc_like.explicitlydeleted = false
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE chp.explicitlydeleted = false AND ci.name = 'Rạch_Giá' AND plc_like.explicitlydeleted = false; -- || 2

-- q8_p019
-- predicates: plc_like.creationdate >= 1351344255062 AND p2.language = 'pt;en'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc_like.creationdate >= 1351344255062 AND p2.language = 'pt;en'; -- || 49

-- q8_p020
-- predicates: chp.explicitlydeleted = false AND c.explicitlydeleted = false
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE chp.explicitlydeleted = false AND c.explicitlydeleted = false; -- || 2158

-- q8_p021
-- predicates: plc_like.creationdate >= 1351344255062 AND p2.language = 'zh;en' AND chp.creationdate BETWEEN 1340327339000 AND 1356461058629
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc_like.creationdate >= 1351344255062 AND p2.language = 'zh;en' AND chp.creationdate BETWEEN 1340327339000 AND 1356461058629; -- || 98

-- q8_p022
-- predicates: plc2.creationdate BETWEEN 1285358988937 AND 1347940587243 AND ci.name = 'Intramuros' AND plc1.explicitlydeleted = false
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc2.creationdate BETWEEN 1285358988937 AND 1347940587243 AND ci.name = 'Intramuros' AND plc1.explicitlydeleted = false; -- || 4

-- q8_p023
-- predicates: plc2.creationdate BETWEEN 1285358988937 AND 1347940587243 AND p2.gender = 'male'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc2.creationdate BETWEEN 1285358988937 AND 1347940587243 AND p2.gender = 'male'; -- || 711

-- q8_p024
-- predicates: plc1.creationdate BETWEEN 1285358988937 AND 1347940587243 AND p1.gender = 'female' AND plc_like.explicitlydeleted = false
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc1.creationdate BETWEEN 1285358988937 AND 1347940587243 AND p1.gender = 'female' AND plc_like.explicitlydeleted = false; -- || 753

-- q8_p025
-- predicates: plc2.creationdate BETWEEN 1285358988937 AND 1347940587243 AND p1.gender = 'female'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc2.creationdate BETWEEN 1285358988937 AND 1347940587243 AND p1.gender = 'female'; -- || 773

-- q8_p026
-- predicates: chp.explicitlydeleted = false AND p1.language = 'es;en'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE chp.explicitlydeleted = false AND p1.language = 'es;en'; -- || 168

-- q8_p027
-- predicates: plc_like.creationdate >= 1351344255062 AND c.creationdate BETWEEN 1340327339000 AND 1356461058629 AND chp.explicitlydeleted = false
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc_like.creationdate >= 1351344255062 AND c.creationdate BETWEEN 1340327339000 AND 1356461058629 AND chp.explicitlydeleted = false; -- || 774

-- q8_p028
-- predicates: plc1.creationdate BETWEEN 1285358988937 AND 1347940587243 AND c.creationdate >= 1350093427787
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc1.creationdate BETWEEN 1285358988937 AND 1347940587243 AND c.creationdate >= 1350093427787; -- || 715

-- q8_p029
-- predicates: plc_like.explicitlydeleted = false AND ci.name = 'Pontianak'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc_like.explicitlydeleted = false AND ci.name = 'Pontianak'; -- || 1

-- q8_p030
-- predicates: chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p1.language = 'zh;en' AND plc2.creationdate BETWEEN 1285358988937 AND 1347940587243
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p1.language = 'zh;en' AND plc2.creationdate BETWEEN 1285358988937 AND 1347940587243; -- || 148

-- q8_p031
-- predicates: plc2.explicitlydeleted = false AND p2.language = 'en'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc2.explicitlydeleted = false AND p2.language = 'en'; -- || 188

-- q8_p032
-- predicates: plc2.creationdate BETWEEN 1285358988937 AND 1347940587243 AND p1.gender = 'female' AND p2.language = 'pt;en'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc2.creationdate BETWEEN 1285358988937 AND 1347940587243 AND p1.gender = 'female' AND p2.language = 'pt;en'; -- || 21

-- q8_p033
-- predicates: plc_like.creationdate >= 1351344255062 AND c.explicitlydeleted = false AND p2.gender = 'male'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc_like.creationdate >= 1351344255062 AND c.explicitlydeleted = false AND p2.gender = 'male'; -- || 467

-- q8_p034
-- predicates: plc_like.explicitlydeleted = false AND c.length BETWEEN 5 AND 79
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc_like.explicitlydeleted = false AND c.length BETWEEN 5 AND 79; -- || 591

-- q8_p035
-- predicates: plc_like.explicitlydeleted = false AND p2.gender = 'female'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc_like.explicitlydeleted = false AND p2.gender = 'female'; -- || 1130

-- q8_p036
-- predicates: plc1.explicitlydeleted = false AND p1.language = 'pt;en'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc1.explicitlydeleted = false AND p1.language = 'pt;en'; -- || 104

-- q8_p037
-- predicates: plc1.explicitlydeleted = false AND ci.name = 'Pontianak'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc1.explicitlydeleted = false AND ci.name = 'Pontianak'; -- || 1

-- q8_p038
-- predicates: chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p2.language = 'es;en' AND plc_like.creationdate >= 1351344255062
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p2.language = 'es;en' AND plc_like.creationdate >= 1351344255062; -- || 61

-- q8_p039
-- predicates: p2.gender = 'female' AND ci.name = 'Intramuros' AND plc2.explicitlydeleted = false
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE p2.gender = 'female' AND ci.name = 'Intramuros' AND plc2.explicitlydeleted = false; -- || 4

-- q8_p040
-- predicates: plc2.explicitlydeleted = false AND p1.gender = 'male'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc2.explicitlydeleted = false AND p1.gender = 'male'; -- || 1016

-- q8_p041
-- predicates: plc2.explicitlydeleted = false AND c.creationdate BETWEEN 1340327339000 AND 1356461058629
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc2.explicitlydeleted = false AND c.creationdate BETWEEN 1340327339000 AND 1356461058629; -- || 1426

-- q8_p042
-- predicates: plc2.explicitlydeleted = false AND p2.language = 'es;en'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc2.explicitlydeleted = false AND p2.language = 'es;en'; -- || 167

-- q8_p043
-- predicates: chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND ci.name = 'Rạch_Giá' AND plc_like.explicitlydeleted = false
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND ci.name = 'Rạch_Giá' AND plc_like.explicitlydeleted = false; -- || 1

-- q8_p044
-- predicates: chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND c.explicitlydeleted = false
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND c.explicitlydeleted = false; -- || 1419

-- q8_p045
-- predicates: chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p1.language = 'pt;en'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p1.language = 'pt;en'; -- || 62

-- q8_p046
-- predicates: chp.explicitlydeleted = false AND p2.language = 'pt;en'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE chp.explicitlydeleted = false AND p2.language = 'pt;en'; -- || 105

-- q8_p047
-- predicates: chp.explicitlydeleted = false AND p1.language = 'en' AND plc2.explicitlydeleted = false
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE chp.explicitlydeleted = false AND p1.language = 'en' AND plc2.explicitlydeleted = false; -- || 183

-- q8_p048
-- predicates: plc1.explicitlydeleted = false AND ci.name = 'Intramuros'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc1.explicitlydeleted = false AND ci.name = 'Intramuros'; -- || 4

-- q8_p049
-- predicates: chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p1.gender = 'male' AND plc2.explicitlydeleted = false
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p1.gender = 'male' AND plc2.explicitlydeleted = false; -- || 669

-- q8_p050
-- predicates: p2.gender = 'male' AND p1.gender = 'male' AND chp.creationdate BETWEEN 1340327339000 AND 1356461058629
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE p2.gender = 'male' AND p1.gender = 'male' AND chp.creationdate BETWEEN 1340327339000 AND 1356461058629; -- || 642

-- q8_p051
-- predicates: plc2.creationdate BETWEEN 1285358988937 AND 1347940587243 AND c.explicitlydeleted = false
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc2.creationdate BETWEEN 1285358988937 AND 1347940587243 AND c.explicitlydeleted = false; -- || 1462

-- q8_p052
-- predicates: plc_like.creationdate >= 1351344255062 AND p2.language = 'en'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc_like.creationdate >= 1351344255062 AND p2.language = 'en'; -- || 95

-- q8_p053
-- predicates: plc1.explicitlydeleted = false AND c.creationdate >= 1350093427787
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc1.explicitlydeleted = false AND c.creationdate >= 1350093427787; -- || 1067

-- q8_p054
-- predicates: plc_like.creationdate >= 1351344255062 AND c.creationdate BETWEEN 1340327339000 AND 1356461058629
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc_like.creationdate >= 1351344255062 AND c.creationdate BETWEEN 1340327339000 AND 1356461058629; -- || 781

-- q8_p055
-- predicates: plc_like.explicitlydeleted = false AND c.creationdate >= 1350093427787 AND plc1.creationdate BETWEEN 1285358988937 AND 1347940587243
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc_like.explicitlydeleted = false AND c.creationdate >= 1350093427787 AND plc1.creationdate BETWEEN 1285358988937 AND 1347940587243; -- || 692

-- q8_p056
-- predicates: plc_like.creationdate >= 1351344255062 AND p2.language = 'pt;en' AND chp.explicitlydeleted = false
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc_like.creationdate >= 1351344255062 AND p2.language = 'pt;en' AND chp.explicitlydeleted = false; -- || 49

-- q8_p057
-- predicates: plc1.explicitlydeleted = false AND c.explicitlydeleted = false
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc1.explicitlydeleted = false AND c.explicitlydeleted = false; -- || 2147

-- q8_p058
-- predicates: chp.explicitlydeleted = false AND p1.gender = 'female' AND c.length BETWEEN 5 AND 79
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE chp.explicitlydeleted = false AND p1.gender = 'female' AND c.length BETWEEN 5 AND 79; -- || 311

-- q8_p059
-- predicates: plc2.explicitlydeleted = false AND p1.gender = 'female' AND p2.language = 'zh;en'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc2.explicitlydeleted = false AND p1.gender = 'female' AND p2.language = 'zh;en'; -- || 139

-- q8_p060
-- predicates: chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND c.length >= 79
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND c.length >= 79; -- || 1103

-- q8_p061
-- predicates: plc1.explicitlydeleted = false AND p1.language = 'zh;en'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc1.explicitlydeleted = false AND p1.language = 'zh;en'; -- || 270

-- q8_p062
-- predicates: plc_like.creationdate >= 1351344255062 AND p2.gender = 'female' AND p1.language = 'pt;en'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc_like.creationdate >= 1351344255062 AND p2.gender = 'female' AND p1.language = 'pt;en'; -- || 21

-- q8_p063
-- predicates: plc2.explicitlydeleted = false AND ci.name = 'Rạch_Giá' AND plc_like.explicitlydeleted = false
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc2.explicitlydeleted = false AND ci.name = 'Rạch_Giá' AND plc_like.explicitlydeleted = false; -- || 2

-- q8_p064
-- predicates: chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p1.language = 'es;en'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p1.language = 'es;en'; -- || 117

-- q8_p065
-- predicates: chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p2.gender = 'male' AND plc2.explicitlydeleted = false
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p2.gender = 'male' AND plc2.explicitlydeleted = false; -- || 671

-- q8_p066
-- predicates: plc1.creationdate BETWEEN 1285358988937 AND 1347940587243 AND p2.language = 'zh;en'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc1.creationdate BETWEEN 1285358988937 AND 1347940587243 AND p2.language = 'zh;en'; -- || 212

-- q8_p067
-- predicates: plc_like.explicitlydeleted = false AND c.creationdate >= 1350093427787
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc_like.explicitlydeleted = false AND c.creationdate >= 1350093427787; -- || 1038

-- q8_p068
-- predicates: plc1.creationdate BETWEEN 1285358988937 AND 1347940587243 AND c.length BETWEEN 5 AND 79 AND ci.name = 'Izmir'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc1.creationdate BETWEEN 1285358988937 AND 1347940587243 AND c.length BETWEEN 5 AND 79 AND ci.name = 'Izmir'; -- || 2

-- q8_p069
-- predicates: plc_like.creationdate >= 1351344255062 AND p1.language = 'pt;en'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc_like.creationdate >= 1351344255062 AND p1.language = 'pt;en'; -- || 49

-- q8_p070
-- predicates: chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND ci.name = 'Intramuros'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE chp.creationdate BETWEEN 1340327339000 AND 1356461058629 AND ci.name = 'Intramuros'; -- || 2

-- q8_p071
-- predicates: plc2.explicitlydeleted = false AND p2.gender = 'male'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc2.explicitlydeleted = false AND p2.gender = 'male'; -- || 1016

-- q8_p072
-- predicates: plc2.explicitlydeleted = false AND p1.language = 'en' AND ci.name = 'Intramuros'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc2.explicitlydeleted = false AND p1.language = 'en' AND ci.name = 'Intramuros'; -- || 4

-- q8_p073
-- predicates: plc1.creationdate BETWEEN 1285358988937 AND 1347940587243 AND c.creationdate BETWEEN 1340327339000 AND 1356461058629 AND chp.creationdate BETWEEN 1340327339000 AND 1356461058629
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc1.creationdate BETWEEN 1285358988937 AND 1347940587243 AND c.creationdate BETWEEN 1340327339000 AND 1356461058629 AND chp.creationdate BETWEEN 1340327339000 AND 1356461058629; -- || 1022

-- q8_p074
-- predicates: chp.explicitlydeleted = false AND p2.gender = 'female'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE chp.explicitlydeleted = false AND p2.gender = 'female'; -- || 1148

-- q8_p075
-- predicates: plc2.creationdate BETWEEN 1285358988937 AND 1347940587243 AND c.length >= 79 AND ci.name = 'Izmir'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc2.creationdate BETWEEN 1285358988937 AND 1347940587243 AND c.length >= 79 AND ci.name = 'Izmir'; -- || 2

-- q8_p076
-- predicates: plc1.explicitlydeleted = false AND c.length BETWEEN 5 AND 79
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc1.explicitlydeleted = false AND c.length BETWEEN 5 AND 79; -- || 600

-- q8_p077
-- predicates: plc_like.creationdate >= 1351344255062 AND p2.gender = 'male' AND chp.creationdate BETWEEN 1340327339000 AND 1356461058629
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc_like.creationdate >= 1351344255062 AND p2.gender = 'male' AND chp.creationdate BETWEEN 1340327339000 AND 1356461058629; -- || 376

-- q8_p078
-- predicates: plc2.creationdate BETWEEN 1285358988937 AND 1347940587243 AND c.explicitlydeleted = false AND plc1.creationdate BETWEEN 1285358988937 AND 1347940587243
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc2.creationdate BETWEEN 1285358988937 AND 1347940587243 AND c.explicitlydeleted = false AND plc1.creationdate BETWEEN 1285358988937 AND 1347940587243; -- || 1432

-- q8_p079
-- predicates: plc2.creationdate BETWEEN 1285358988937 AND 1347940587243 AND p2.gender = 'male' AND c.length >= 79
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc2.creationdate BETWEEN 1285358988937 AND 1347940587243 AND p2.gender = 'male' AND c.length >= 79; -- || 536

-- q8_p080
-- predicates: plc_like.creationdate >= 1351344255062 AND c.length >= 79
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc_like.creationdate >= 1351344255062 AND c.length >= 79; -- || 777

-- q8_p081
-- predicates: plc2.creationdate BETWEEN 1285358988937 AND 1347940587243 AND p2.language = 'es;en'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc2.creationdate BETWEEN 1285358988937 AND 1347940587243 AND p2.language = 'es;en'; -- || 91

-- q8_p082
-- predicates: plc_like.creationdate >= 1351344255062 AND p1.gender = 'female'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc_like.creationdate >= 1351344255062 AND p1.gender = 'female'; -- || 506

-- q8_p083
-- predicates: plc1.creationdate BETWEEN 1285358988937 AND 1347940587243 AND p2.gender = 'male'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc1.creationdate BETWEEN 1285358988937 AND 1347940587243 AND p2.gender = 'male'; -- || 706

-- q8_p084
-- predicates: plc1.explicitlydeleted = false AND c.creationdate BETWEEN 1340327339000 AND 1356461058629
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc1.explicitlydeleted = false AND c.creationdate BETWEEN 1340327339000 AND 1356461058629; -- || 1426

-- q8_p085
-- predicates: plc2.creationdate BETWEEN 1285358988937 AND 1347940587243 AND ci.name = 'Pontianak' AND c.explicitlydeleted = false
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc2.creationdate BETWEEN 1285358988937 AND 1347940587243 AND ci.name = 'Pontianak' AND c.explicitlydeleted = false; -- || 1

-- q8_p086
-- predicates: chp.explicitlydeleted = false AND p1.gender = 'male'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE chp.explicitlydeleted = false AND p1.gender = 'male'; -- || 1011

-- q8_p087
-- predicates: plc1.creationdate BETWEEN 1285358988937 AND 1347940587243 AND p2.gender = 'female'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc1.creationdate BETWEEN 1285358988937 AND 1347940587243 AND p2.gender = 'female'; -- || 780

-- q8_p088
-- predicates: plc_like.creationdate >= 1351344255062 AND c.length >= 79 AND plc2.explicitlydeleted = false
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc_like.creationdate >= 1351344255062 AND c.length >= 79 AND plc2.explicitlydeleted = false; -- || 775

-- q8_p089
-- predicates: plc1.explicitlydeleted = false AND p1.gender = 'female' AND plc2.explicitlydeleted = false
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc1.explicitlydeleted = false AND p1.gender = 'female' AND plc2.explicitlydeleted = false; -- || 1156

-- q8_p090
-- predicates: plc_like.explicitlydeleted = false AND p1.gender = 'female' AND plc1.explicitlydeleted = false
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc_like.explicitlydeleted = false AND p1.gender = 'female' AND plc1.explicitlydeleted = false; -- || 1125

-- q8_p091
-- predicates: plc_like.creationdate >= 1351344255062 AND p1.language = 'pt;en' AND plc1.explicitlydeleted = false
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc_like.creationdate >= 1351344255062 AND p1.language = 'pt;en' AND plc1.explicitlydeleted = false; -- || 49

-- q8_p092
-- predicates: plc_like.explicitlydeleted = false AND c.length >= 79
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc_like.explicitlydeleted = false AND c.length >= 79; -- || 1624

-- q8_p093
-- predicates: plc1.explicitlydeleted = false AND p1.language = 'en'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc1.explicitlydeleted = false AND p1.language = 'en'; -- || 185

-- q8_p094
-- predicates: plc2.creationdate BETWEEN 1285358988937 AND 1347940587243 AND p2.gender = 'female'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc2.creationdate BETWEEN 1285358988937 AND 1347940587243 AND p2.gender = 'female'; -- || 768

-- q8_p095
-- predicates: chp.explicitlydeleted = false AND p2.language = 'en'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE chp.explicitlydeleted = false AND p2.language = 'en'; -- || 187

-- q8_p096
-- predicates: chp.explicitlydeleted = false AND p1.language = 'en'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE chp.explicitlydeleted = false AND p1.language = 'en'; -- || 184

-- q8_p097
-- predicates: plc_like.creationdate >= 1351344255062 AND c.length BETWEEN 5 AND 79
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc_like.creationdate >= 1351344255062 AND c.length BETWEEN 5 AND 79; -- || 244

-- q8_p098
-- predicates: plc2.creationdate BETWEEN 1285358988937 AND 1347940587243 AND p1.language = 'en'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc2.creationdate BETWEEN 1285358988937 AND 1347940587243 AND p1.language = 'en'; -- || 131

-- q8_p099
-- predicates: plc2.creationdate BETWEEN 1285358988937 AND 1347940587243 AND p1.gender = 'female' AND plc1.explicitlydeleted = false
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc2.creationdate BETWEEN 1285358988937 AND 1347940587243 AND p1.gender = 'female' AND plc1.explicitlydeleted = false; -- || 772

-- q8_p100
-- predicates: plc_like.explicitlydeleted = false AND p2.language = 'pt;en'
SELECT
  -- 点表
  p1.*,
  c.*,
  p2.*,
  ci.*,

  -- 边表
  plc_like.*,
  chp.*,
  plc1.*,
  plc2.*
FROM person AS p1
JOIN person_likes_comment AS plc_like
  ON plc_like.personid = p1.id
JOIN comment AS c
  ON c.id = plc_like.commentid
JOIN comment_hascreator_person AS chp
  ON chp.commentid = c.id
JOIN person AS p2
  ON p2.id = chp.personid
JOIN person_islocatedin_city AS plc1
  ON plc1.personid = p1.id
JOIN person_islocatedin_city AS plc2
  ON plc2.personid = p2.id
 AND plc2.cityid = plc1.cityid
JOIN city AS ci
  ON ci.id = plc1.cityid
WHERE plc_like.explicitlydeleted = false AND p2.language = 'pt;en'; -- || 101

