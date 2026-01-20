-- Q10 base true_cardinality: 20
-- Generated with seed=20260115

-- q10_p001
-- predicates: chp2.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p2.gender = 'female' AND plc2.creationdate BETWEEN 1343776564773 AND 1356763691590
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE chp2.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p2.gender = 'female' AND plc2.creationdate BETWEEN 1343776564773 AND 1356763691590; -- || 4

-- q10_p002
-- predicates: chp1.explicitlydeleted = false AND p2.gender = 'female'
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE chp1.explicitlydeleted = false AND p2.gender = 'female'; -- || 9

-- q10_p003
-- predicates: rcc.creationdate >= 1351930981563 AND c2.creationdate BETWEEN 1340327339000 AND 1356461058629
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE rcc.creationdate >= 1351930981563 AND c2.creationdate BETWEEN 1340327339000 AND 1356461058629; -- || 7

-- q10_p004
-- predicates: chp2.creationdate BETWEEN 1340327339000 AND 1356461058629 AND c1.creationdate >= 1350093427787 AND plc2.explicitlydeleted = false
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE chp2.creationdate BETWEEN 1340327339000 AND 1356461058629 AND c1.creationdate >= 1350093427787 AND plc2.explicitlydeleted = false; -- || 8

-- q10_p005
-- predicates: chp2.explicitlydeleted = false AND c1.explicitlydeleted = false
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE chp2.explicitlydeleted = false AND c1.explicitlydeleted = false; -- || 20

-- q10_p006
-- predicates: chp1.explicitlydeleted = false AND p2.gender = 'male'
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE chp1.explicitlydeleted = false AND p2.gender = 'male'; -- || 11

-- q10_p007
-- predicates: chp2.explicitlydeleted = false AND p1.language = 'zh;en'
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE chp2.explicitlydeleted = false AND p1.language = 'zh;en'; -- || 4

-- q10_p008
-- predicates: plc2.creationdate >= 1351344255062 AND p2.gender = 'male' AND c2.length >= 79
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE plc2.creationdate >= 1351344255062 AND p2.gender = 'male' AND c2.length >= 79; -- || 6

-- q10_p009
-- predicates: rcc.explicitlydeleted = false AND c2.length >= 79 AND plc1.creationdate BETWEEN 1343776564773 AND 1356763691590
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE rcc.explicitlydeleted = false AND c2.length >= 79 AND plc1.creationdate BETWEEN 1343776564773 AND 1356763691590; -- || 10

-- q10_p010
-- predicates: plc2.creationdate BETWEEN 1343776564773 AND 1356763691590 AND c1.creationdate >= 1350093427787 AND rcc.creationdate BETWEEN 1345196662740 AND 1356577357333
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE plc2.creationdate BETWEEN 1343776564773 AND 1356763691590 AND c1.creationdate >= 1350093427787 AND rcc.creationdate BETWEEN 1345196662740 AND 1356577357333; -- || 8

-- q10_p011
-- predicates: plc1.creationdate BETWEEN 1343776564773 AND 1356763691590 AND p2.language = 'en'
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE plc1.creationdate BETWEEN 1343776564773 AND 1356763691590 AND p2.language = 'en'; -- || 3

-- q10_p012
-- predicates: rcc.creationdate BETWEEN 1345196662740 AND 1356577357333 AND c2.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p1.gender = 'female'
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE rcc.creationdate BETWEEN 1345196662740 AND 1356577357333 AND c2.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p1.gender = 'female'; -- || 6

-- q10_p013
-- predicates: rcc.explicitlydeleted = false AND c1.creationdate >= 1350093427787 AND plc2.creationdate BETWEEN 1343776564773 AND 1356763691590
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE rcc.explicitlydeleted = false AND c1.creationdate >= 1350093427787 AND plc2.creationdate BETWEEN 1343776564773 AND 1356763691590; -- || 8

-- q10_p014
-- predicates: c1.length >= 79 AND p2.language = 'pt;en' AND plc2.creationdate BETWEEN 1343776564773 AND 1356763691590
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE c1.length >= 79 AND p2.language = 'pt;en' AND plc2.creationdate BETWEEN 1343776564773 AND 1356763691590; -- || 1

-- q10_p015
-- predicates: rcc.creationdate BETWEEN 1345196662740 AND 1356577357333 AND c1.length >= 79 AND c2.length BETWEEN 5 AND 79
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE rcc.creationdate BETWEEN 1345196662740 AND 1356577357333 AND c1.length >= 79 AND c2.length BETWEEN 5 AND 79; -- || 3

-- q10_p016
-- predicates: plc1.explicitlydeleted = false AND p1.language = 'pt;en'
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE plc1.explicitlydeleted = false AND p1.language = 'pt;en'; -- || 1

-- q10_p017
-- predicates: plc1.creationdate BETWEEN 1343776564773 AND 1356763691590 AND c2.creationdate BETWEEN 1340327339000 AND 1356461058629
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE plc1.creationdate BETWEEN 1343776564773 AND 1356763691590 AND c2.creationdate BETWEEN 1340327339000 AND 1356461058629; -- || 11

-- q10_p018
-- predicates: chp2.explicitlydeleted = false AND p1.gender = 'female'
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE chp2.explicitlydeleted = false AND p1.gender = 'female'; -- || 11

-- q10_p019
-- predicates: chp2.explicitlydeleted = false AND p1.language = 'es;en'
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE chp2.explicitlydeleted = false AND p1.language = 'es;en'; -- || 1

-- q10_p020
-- predicates: p2.gender = 'female' AND c1.length BETWEEN 5 AND 79
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE p2.gender = 'female' AND c1.length BETWEEN 5 AND 79; -- || 2

-- q10_p021
-- predicates: chp1.explicitlydeleted = false AND p2.language = 'zh;en'
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE chp1.explicitlydeleted = false AND p2.language = 'zh;en'; -- || 5

-- q10_p022
-- predicates: rcc.explicitlydeleted = false AND c2.length BETWEEN 5 AND 79
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE rcc.explicitlydeleted = false AND c2.length BETWEEN 5 AND 79; -- || 6

-- q10_p023
-- predicates: plc1.explicitlydeleted = false AND p1.gender = 'female' AND c1.length BETWEEN 5 AND 79
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE plc1.explicitlydeleted = false AND p1.gender = 'female' AND c1.length BETWEEN 5 AND 79; -- || 3

-- q10_p024
-- predicates: chp2.creationdate BETWEEN 1340327339000 AND 1356461058629 AND c2.explicitlydeleted = false
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE chp2.creationdate BETWEEN 1340327339000 AND 1356461058629 AND c2.explicitlydeleted = false; -- || 12

-- q10_p025
-- predicates: chp1.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p2.language = 'pt;en'
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE chp1.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p2.language = 'pt;en'; -- || 1

-- q10_p026
-- predicates: chp1.explicitlydeleted = false AND c1.creationdate >= 1350093427787 AND c2.length BETWEEN 5 AND 79
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE chp1.explicitlydeleted = false AND c1.creationdate >= 1350093427787 AND c2.length BETWEEN 5 AND 79; -- || 3

-- q10_p027
-- predicates: chp1.creationdate BETWEEN 1340327339000 AND 1356461058629 AND c2.length BETWEEN 5 AND 79
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE chp1.creationdate BETWEEN 1340327339000 AND 1356461058629 AND c2.length BETWEEN 5 AND 79; -- || 4

-- q10_p028
-- predicates: rcc.creationdate BETWEEN 1345196662740 AND 1356577357333 AND c1.creationdate BETWEEN 1340327339000 AND 1356461058629
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE rcc.creationdate BETWEEN 1345196662740 AND 1356577357333 AND c1.creationdate BETWEEN 1340327339000 AND 1356461058629; -- || 12

-- q10_p029
-- predicates: plc2.creationdate >= 1351344255062 AND c2.creationdate BETWEEN 1340327339000 AND 1356461058629
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE plc2.creationdate >= 1351344255062 AND c2.creationdate BETWEEN 1340327339000 AND 1356461058629; -- || 7

-- q10_p030
-- predicates: plc1.creationdate BETWEEN 1343776564773 AND 1356763691590 AND c1.creationdate BETWEEN 1340327339000 AND 1356461058629
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE plc1.creationdate BETWEEN 1343776564773 AND 1356763691590 AND c1.creationdate BETWEEN 1340327339000 AND 1356461058629; -- || 13

-- q10_p031
-- predicates: chp2.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p2.gender = 'female' AND plc2.creationdate >= 1351344255062
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE chp2.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p2.gender = 'female' AND plc2.creationdate >= 1351344255062; -- || 2

-- q10_p032
-- predicates: plc2.explicitlydeleted = false AND c1.length >= 79 AND c2.explicitlydeleted = false
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE plc2.explicitlydeleted = false AND c1.length >= 79 AND c2.explicitlydeleted = false; -- || 17

-- q10_p033
-- predicates: rcc.explicitlydeleted = false AND p1.language = 'es;en' AND c1.creationdate >= 1350093427787
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE rcc.explicitlydeleted = false AND p1.language = 'es;en' AND c1.creationdate >= 1350093427787; -- || 1

-- q10_p034
-- predicates: plc1.creationdate BETWEEN 1343776564773 AND 1356763691590 AND c1.length BETWEEN 5 AND 79
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE plc1.creationdate BETWEEN 1343776564773 AND 1356763691590 AND c1.length BETWEEN 5 AND 79; -- || 2

-- q10_p035
-- predicates: plc1.creationdate >= 1351344255062 AND c1.length BETWEEN 5 AND 79
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE plc1.creationdate >= 1351344255062 AND c1.length BETWEEN 5 AND 79; -- || 1

-- q10_p036
-- predicates: chp1.creationdate BETWEEN 1340327339000 AND 1356461058629 AND c1.explicitlydeleted = false AND p2.gender = 'male'
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE chp1.creationdate BETWEEN 1340327339000 AND 1356461058629 AND c1.explicitlydeleted = false AND p2.gender = 'male'; -- || 9

-- q10_p037
-- predicates: chp2.creationdate BETWEEN 1340327339000 AND 1356461058629 AND c1.creationdate BETWEEN 1340327339000 AND 1356461058629
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE chp2.creationdate BETWEEN 1340327339000 AND 1356461058629 AND c1.creationdate BETWEEN 1340327339000 AND 1356461058629; -- || 12

-- q10_p038
-- predicates: plc1.creationdate BETWEEN 1343776564773 AND 1356763691590 AND c1.length >= 79
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE plc1.creationdate BETWEEN 1343776564773 AND 1356763691590 AND c1.length >= 79; -- || 12

-- q10_p039
-- predicates: plc1.explicitlydeleted = false AND c1.explicitlydeleted = false AND plc2.explicitlydeleted = false
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE plc1.explicitlydeleted = false AND c1.explicitlydeleted = false AND plc2.explicitlydeleted = false; -- || 19

-- q10_p040
-- predicates: plc2.creationdate BETWEEN 1343776564773 AND 1356763691590 AND p1.language = 'zh;en'
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE plc2.creationdate BETWEEN 1343776564773 AND 1356763691590 AND p1.language = 'zh;en'; -- || 2

-- q10_p041
-- predicates: rcc.creationdate BETWEEN 1345196662740 AND 1356577357333 AND c2.creationdate BETWEEN 1340327339000 AND 1356461058629
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE rcc.creationdate BETWEEN 1345196662740 AND 1356577357333 AND c2.creationdate BETWEEN 1340327339000 AND 1356461058629; -- || 10

-- q10_p042
-- predicates: chp1.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p1.gender = 'female'
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE chp1.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p1.gender = 'female'; -- || 7

-- q10_p043
-- predicates: chp1.explicitlydeleted = false AND c2.length BETWEEN 5 AND 79
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE chp1.explicitlydeleted = false AND c2.length BETWEEN 5 AND 79; -- || 6

-- q10_p044
-- predicates: chp2.explicitlydeleted = false AND p2.language = 'zh;en'
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE chp2.explicitlydeleted = false AND p2.language = 'zh;en'; -- || 5

-- q10_p045
-- predicates: plc1.explicitlydeleted = false AND p2.gender = 'male' AND rcc.creationdate BETWEEN 1345196662740 AND 1356577357333
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE plc1.explicitlydeleted = false AND p2.gender = 'male' AND rcc.creationdate BETWEEN 1345196662740 AND 1356577357333; -- || 8

-- q10_p046
-- predicates: chp1.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p1.language = 'pt;en'
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE chp1.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p1.language = 'pt;en'; -- || 1

-- q10_p047
-- predicates: chp2.creationdate BETWEEN 1340327339000 AND 1356461058629 AND c2.creationdate BETWEEN 1340327339000 AND 1356461058629
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE chp2.creationdate BETWEEN 1340327339000 AND 1356461058629 AND c2.creationdate BETWEEN 1340327339000 AND 1356461058629; -- || 12

-- q10_p048
-- predicates: rcc.creationdate BETWEEN 1345196662740 AND 1356577357333 AND p1.gender = 'female' AND chp2.creationdate BETWEEN 1340327339000 AND 1356461058629
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE rcc.creationdate BETWEEN 1345196662740 AND 1356577357333 AND p1.gender = 'female' AND chp2.creationdate BETWEEN 1340327339000 AND 1356461058629; -- || 6

-- q10_p049
-- predicates: p2.gender = 'male' AND c1.explicitlydeleted = false AND p1.gender = 'female'
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE p2.gender = 'male' AND c1.explicitlydeleted = false AND p1.gender = 'female'; -- || 7

-- q10_p050
-- predicates: rcc.creationdate >= 1351930981563 AND p2.gender = 'female'
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE rcc.creationdate >= 1351930981563 AND p2.gender = 'female'; -- || 4

-- q10_p051
-- predicates: rcc.creationdate >= 1351930981563 AND p2.language = 'en' AND plc2.creationdate >= 1351344255062
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE rcc.creationdate >= 1351930981563 AND p2.language = 'en' AND plc2.creationdate >= 1351344255062; -- || 3

-- q10_p052
-- predicates: plc1.creationdate >= 1351344255062 AND p2.gender = 'male' AND rcc.creationdate >= 1351930981563
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE plc1.creationdate >= 1351344255062 AND p2.gender = 'male' AND rcc.creationdate >= 1351930981563; -- || 6

-- q10_p053
-- predicates: chp1.explicitlydeleted = false AND c1.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p2.language = 'zh;en'
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE chp1.explicitlydeleted = false AND c1.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p2.language = 'zh;en'; -- || 1

-- q10_p054
-- predicates: chp1.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p1.language = 'de;en'
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE chp1.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p1.language = 'de;en'; -- || 2

-- q10_p055
-- predicates: plc1.explicitlydeleted = false AND c2.explicitlydeleted = false
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE plc1.explicitlydeleted = false AND c2.explicitlydeleted = false; -- || 19

-- q10_p056
-- predicates: plc1.explicitlydeleted = false AND p1.gender = 'female'
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE plc1.explicitlydeleted = false AND p1.gender = 'female'; -- || 10

-- q10_p057
-- predicates: rcc.explicitlydeleted = false AND p2.language = 'zh;en' AND plc1.creationdate >= 1351344255062
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE rcc.explicitlydeleted = false AND p2.language = 'zh;en' AND plc1.creationdate >= 1351344255062; -- || 3

-- q10_p058
-- predicates: chp2.creationdate BETWEEN 1340327339000 AND 1356461058629 AND c1.length BETWEEN 5 AND 79
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE chp2.creationdate BETWEEN 1340327339000 AND 1356461058629 AND c1.length BETWEEN 5 AND 79; -- || 1

-- q10_p059
-- predicates: chp1.explicitlydeleted = false AND c2.explicitlydeleted = false
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE chp1.explicitlydeleted = false AND c2.explicitlydeleted = false; -- || 20

-- q10_p060
-- predicates: plc2.explicitlydeleted = false AND c1.creationdate BETWEEN 1340327339000 AND 1356461058629 AND plc1.creationdate >= 1351344255062
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE plc2.explicitlydeleted = false AND c1.creationdate BETWEEN 1340327339000 AND 1356461058629 AND plc1.creationdate >= 1351344255062; -- || 7

-- q10_p061
-- predicates: rcc.creationdate >= 1351930981563 AND p1.language = 'es;en' AND c1.length >= 79
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE rcc.creationdate >= 1351930981563 AND p1.language = 'es;en' AND c1.length >= 79; -- || 1

-- q10_p062
-- predicates: chp1.explicitlydeleted = false AND p1.language = 'es;en'
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE chp1.explicitlydeleted = false AND p1.language = 'es;en'; -- || 1

-- q10_p063
-- predicates: plc2.explicitlydeleted = false AND c1.length >= 79
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE plc2.explicitlydeleted = false AND c1.length >= 79; -- || 17

-- q10_p064
-- predicates: chp1.explicitlydeleted = false AND c1.creationdate BETWEEN 1340327339000 AND 1356461058629 AND rcc.creationdate >= 1351930981563
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE chp1.explicitlydeleted = false AND c1.creationdate BETWEEN 1340327339000 AND 1356461058629 AND rcc.creationdate >= 1351930981563; -- || 7

-- q10_p065
-- predicates: chp2.explicitlydeleted = false AND p1.language = 'es;en' AND plc2.explicitlydeleted = false
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE chp2.explicitlydeleted = false AND p1.language = 'es;en' AND plc2.explicitlydeleted = false; -- || 1

-- q10_p066
-- predicates: rcc.explicitlydeleted = false AND p2.language = 'en' AND plc1.explicitlydeleted = false
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE rcc.explicitlydeleted = false AND p2.language = 'en' AND plc1.explicitlydeleted = false; -- || 3

-- q10_p067
-- predicates: p1.language = 'es;en' AND p2.gender = 'female'
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE p1.language = 'es;en' AND p2.gender = 'female'; -- || 1

-- q10_p068
-- predicates: rcc.explicitlydeleted = false AND p1.gender = 'female' AND chp2.explicitlydeleted = false
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE rcc.explicitlydeleted = false AND p1.gender = 'female' AND chp2.explicitlydeleted = false; -- || 11

-- q10_p069
-- predicates: plc2.explicitlydeleted = false AND c2.creationdate >= 1350093427787
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE plc2.explicitlydeleted = false AND c2.creationdate >= 1350093427787; -- || 11

-- q10_p070
-- predicates: rcc.creationdate >= 1351930981563 AND c1.length BETWEEN 5 AND 79
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE rcc.creationdate >= 1351930981563 AND c1.length BETWEEN 5 AND 79; -- || 1

-- q10_p071
-- predicates: plc1.explicitlydeleted = false AND c1.length BETWEEN 5 AND 79
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE plc1.explicitlydeleted = false AND c1.length BETWEEN 5 AND 79; -- || 3

-- q10_p072
-- predicates: plc1.creationdate >= 1351344255062 AND p2.gender = 'male' AND p1.gender = 'male'
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE plc1.creationdate >= 1351344255062 AND p2.gender = 'male' AND p1.gender = 'male'; -- || 2

-- q10_p073
-- predicates: plc2.creationdate BETWEEN 1343776564773 AND 1356763691590 AND c1.explicitlydeleted = false AND p2.gender = 'female'
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE plc2.creationdate BETWEEN 1343776564773 AND 1356763691590 AND c1.explicitlydeleted = false AND p2.gender = 'female'; -- || 4

-- q10_p074
-- predicates: plc2.explicitlydeleted = false AND c2.length >= 79
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE plc2.explicitlydeleted = false AND c2.length >= 79; -- || 16

-- q10_p075
-- predicates: plc2.creationdate >= 1351344255062 AND c1.creationdate BETWEEN 1340327339000 AND 1356461058629 AND rcc.creationdate >= 1351930981563
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE plc2.creationdate >= 1351344255062 AND c1.creationdate BETWEEN 1340327339000 AND 1356461058629 AND rcc.creationdate >= 1351930981563; -- || 7

-- q10_p076
-- predicates: plc2.creationdate >= 1351344255062 AND c1.length >= 79
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE plc2.creationdate >= 1351344255062 AND c1.length >= 79; -- || 9

-- q10_p077
-- predicates: chp1.explicitlydeleted = false AND c1.creationdate BETWEEN 1340327339000 AND 1356461058629
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE chp1.explicitlydeleted = false AND c1.creationdate BETWEEN 1340327339000 AND 1356461058629; -- || 14

-- q10_p078
-- predicates: chp2.explicitlydeleted = false AND c2.length BETWEEN 5 AND 79 AND rcc.creationdate BETWEEN 1345196662740 AND 1356577357333
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE chp2.explicitlydeleted = false AND c2.length BETWEEN 5 AND 79 AND rcc.creationdate BETWEEN 1345196662740 AND 1356577357333; -- || 4

-- q10_p079
-- predicates: plc1.creationdate BETWEEN 1343776564773 AND 1356763691590 AND p2.language = 'pt;en'
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE plc1.creationdate BETWEEN 1343776564773 AND 1356763691590 AND p2.language = 'pt;en'; -- || 1

-- q10_p080
-- predicates: plc2.creationdate >= 1351344255062 AND p2.language = 'en'
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE plc2.creationdate >= 1351344255062 AND p2.language = 'en'; -- || 3

-- q10_p081
-- predicates: rcc.creationdate >= 1351930981563 AND c1.length >= 79 AND chp2.creationdate BETWEEN 1340327339000 AND 1356461058629
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE rcc.creationdate >= 1351930981563 AND c1.length >= 79 AND chp2.creationdate BETWEEN 1340327339000 AND 1356461058629; -- || 7

-- q10_p082
-- predicates: c1.explicitlydeleted = false AND c2.explicitlydeleted = false
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE c1.explicitlydeleted = false AND c2.explicitlydeleted = false; -- || 20

-- q10_p083
-- predicates: chp1.creationdate BETWEEN 1340327339000 AND 1356461058629 AND c2.creationdate >= 1350093427787
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE chp1.creationdate BETWEEN 1340327339000 AND 1356461058629 AND c2.creationdate >= 1350093427787; -- || 8

-- q10_p084
-- predicates: chp2.explicitlydeleted = false AND p1.language = 'de;en'
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE chp2.explicitlydeleted = false AND p1.language = 'de;en'; -- || 3

-- q10_p085
-- predicates: plc2.explicitlydeleted = false AND c2.length >= 79 AND plc1.explicitlydeleted = false
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE plc2.explicitlydeleted = false AND c2.length >= 79 AND plc1.explicitlydeleted = false; -- || 15

-- q10_p086
-- predicates: chp1.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p1.language = 'zh;en'
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE chp1.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p1.language = 'zh;en'; -- || 3

-- q10_p087
-- predicates: rcc.creationdate BETWEEN 1345196662740 AND 1356577357333 AND c1.explicitlydeleted = false AND p1.language = 'zh;en'
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE rcc.creationdate BETWEEN 1345196662740 AND 1356577357333 AND c1.explicitlydeleted = false AND p1.language = 'zh;en'; -- || 3

-- q10_p088
-- predicates: plc1.explicitlydeleted = false AND c2.length BETWEEN 5 AND 79
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE plc1.explicitlydeleted = false AND c2.length BETWEEN 5 AND 79; -- || 6

-- q10_p089
-- predicates: plc2.explicitlydeleted = false AND c2.creationdate BETWEEN 1340327339000 AND 1356461058629
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE plc2.explicitlydeleted = false AND c2.creationdate BETWEEN 1340327339000 AND 1356461058629; -- || 12

-- q10_p090
-- predicates: chp2.explicitlydeleted = false AND p1.gender = 'female' AND c1.length BETWEEN 5 AND 79
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE chp2.explicitlydeleted = false AND p1.gender = 'female' AND c1.length BETWEEN 5 AND 79; -- || 4

-- q10_p091
-- predicates: plc2.creationdate >= 1351344255062 AND p1.gender = 'male' AND c2.length >= 79
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE plc2.creationdate >= 1351344255062 AND p1.gender = 'male' AND c2.length >= 79; -- || 3

-- q10_p092
-- predicates: chp1.creationdate BETWEEN 1340327339000 AND 1356461058629 AND c2.creationdate >= 1350093427787 AND plc1.explicitlydeleted = false
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE chp1.creationdate BETWEEN 1340327339000 AND 1356461058629 AND c2.creationdate >= 1350093427787 AND plc1.explicitlydeleted = false; -- || 8

-- q10_p093
-- predicates: plc2.creationdate BETWEEN 1343776564773 AND 1356763691590 AND p2.language = 'pt;en' AND c1.creationdate BETWEEN 1340327339000 AND 1356461058629
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE plc2.creationdate BETWEEN 1343776564773 AND 1356763691590 AND p2.language = 'pt;en' AND c1.creationdate BETWEEN 1340327339000 AND 1356461058629; -- || 1

-- q10_p094
-- predicates: chp2.explicitlydeleted = false AND p2.gender = 'female' AND plc2.explicitlydeleted = false
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE chp2.explicitlydeleted = false AND p2.gender = 'female' AND plc2.explicitlydeleted = false; -- || 9

-- q10_p095
-- predicates: chp2.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p1.language = 'pt;en'
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE chp2.creationdate BETWEEN 1340327339000 AND 1356461058629 AND p1.language = 'pt;en'; -- || 1

-- q10_p096
-- predicates: plc1.explicitlydeleted = false AND c1.explicitlydeleted = false
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE plc1.explicitlydeleted = false AND c1.explicitlydeleted = false; -- || 19

-- q10_p097
-- predicates: chp1.explicitlydeleted = false AND c1.creationdate >= 1350093427787
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE chp1.explicitlydeleted = false AND c1.creationdate >= 1350093427787; -- || 12

-- q10_p098
-- predicates: plc1.explicitlydeleted = false AND p1.language = 'de;en'
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE plc1.explicitlydeleted = false AND p1.language = 'de;en'; -- || 3

-- q10_p099
-- predicates: chp2.explicitlydeleted = false AND p2.language = 'en'
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE chp2.explicitlydeleted = false AND p2.language = 'en'; -- || 3

-- q10_p100
-- predicates: plc2.creationdate BETWEEN 1343776564773 AND 1356763691590 AND c2.length >= 79
SELECT
  -- 点表
  p1.*,
  c1.*,
  p2.*,
  c2.*,

  -- 边表
  plc1.*,
  chp1.*,
  rcc.*,
  plc2.*,
  chp2.*
FROM person AS p1
JOIN person_likes_comment AS plc1
  ON plc1.personid = p1.id
JOIN comment AS c1
  ON c1.id = plc1.commentid
JOIN comment_hascreator_person AS chp1
  ON chp1.commentid = c1.id
JOIN person AS p2
  ON p2.id = chp1.personid
JOIN comment_replyof_comment AS rcc
  ON rcc.comment1id = c1.id
JOIN comment AS c2
  ON c2.id = rcc.comment2id
JOIN person_likes_comment AS plc2
  ON plc2.personid = p2.id
 AND plc2.commentid = c2.id
JOIN comment_hascreator_person AS chp2
  ON chp2.commentid = c2.id
 AND chp2.personid = p1.id
WHERE plc2.creationdate BETWEEN 1343776564773 AND 1356763691590 AND c2.length >= 79; -- || 9

