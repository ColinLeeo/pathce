-- Q11 base true_cardinality: 182818
-- Generated with seed=20260115

-- q11_p001
-- predicates: pkp.src BETWEEN 252947 AND 253760 AND f.title = 'Wall of Rahul Khan' AND fhm1.explicitlydeleted = false
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE pkp.src BETWEEN 252947 AND 253760 AND f.title = 'Wall of Rahul Khan' AND fhm1.explicitlydeleted = false; -- || 1

-- q11_p002
-- predicates: plp1.explicitlydeleted = false AND p2.gender = 'female' AND fhm1.creationdate BETWEEN 1321789849404 AND 1354857790702
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE plp1.explicitlydeleted = false AND p2.gender = 'female' AND fhm1.creationdate BETWEEN 1321789849404 AND 1354857790702; -- || 70108

-- q11_p003
-- predicates: fhm2.explicitlydeleted = false AND po.creationdate BETWEEN 1308052370341 AND 1352177414840
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fhm2.explicitlydeleted = false AND po.creationdate BETWEEN 1308052370341 AND 1352177414840; -- || 136074

-- q11_p004
-- predicates: plp2.creationdate BETWEEN 1321581133136 AND 1354960011366 AND p1.language = 'en' AND plp1.explicitlydeleted = false
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE plp2.creationdate BETWEEN 1321581133136 AND 1354960011366 AND p1.language = 'en' AND plp1.explicitlydeleted = false; -- || 10505

-- q11_p005
-- predicates: fcp.explicitlydeleted = false AND p2.language = 'es;en'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fcp.explicitlydeleted = false AND p2.language = 'es;en'; -- || 10959

-- q11_p006
-- predicates: fcp.explicitlydeleted = false AND p1.language = 'en' AND f.title = 'Wall of Rahul Khan'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fcp.explicitlydeleted = false AND p1.language = 'en' AND f.title = 'Wall of Rahul Khan'; -- || 1

-- q11_p007
-- predicates: pkp.dst <= 253343 AND po.creationdate BETWEEN 1308052370341 AND 1352177414840 AND p1.language = 'pt;en'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE pkp.dst <= 253343 AND po.creationdate BETWEEN 1308052370341 AND 1352177414840 AND p1.language = 'pt;en'; -- || 4046

-- q11_p008
-- predicates: p1.language = 'de;en' AND po.length BETWEEN 0 AND 0 AND pkp.dst BETWEEN 252947 AND 253760
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE p1.language = 'de;en' AND po.length BETWEEN 0 AND 0 AND pkp.dst BETWEEN 252947 AND 253760; -- || 3796

-- q11_p009
-- predicates: fhm1.creationdate BETWEEN 1321789849404 AND 1354857790702 AND p1.language = 'es;en'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fhm1.creationdate BETWEEN 1321789849404 AND 1354857790702 AND p1.language = 'es;en'; -- || 7551

-- q11_p010
-- predicates: p2.language = 'pt;en' AND f.creationdate BETWEEN 1305589168419 AND 1351610285871 AND po.length <= 0
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE p2.language = 'pt;en' AND f.creationdate BETWEEN 1305589168419 AND 1351610285871 AND po.length <= 0; -- || 3814

-- q11_p011
-- predicates: pkp.src BETWEEN 252947 AND 253760 AND po.length BETWEEN 0 AND 0 AND plp1.explicitlydeleted = false
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE pkp.src BETWEEN 252947 AND 253760 AND po.length BETWEEN 0 AND 0 AND plp1.explicitlydeleted = false; -- || 60277

-- q11_p012
-- predicates: pkp.dst <= 253343 AND p2.language = 'zh;en' AND fhm1.explicitlydeleted = false
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE pkp.dst <= 253343 AND p2.language = 'zh;en' AND fhm1.explicitlydeleted = false; -- || 13565

-- q11_p013
-- predicates: fhm2.explicitlydeleted = false AND po.language = 'en' AND plp1.explicitlydeleted = false
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fhm2.explicitlydeleted = false AND po.language = 'en' AND plp1.explicitlydeleted = false; -- || 45584

-- q11_p014
-- predicates: pkp.dst BETWEEN 252947 AND 253760 AND p1.language = 'de;en'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE pkp.dst BETWEEN 252947 AND 253760 AND p1.language = 'de;en'; -- || 6711

-- q11_p015
-- predicates: fhm1.creationdate BETWEEN 1321789849404 AND 1354857790702 AND p2.gender = 'female'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fhm1.creationdate BETWEEN 1321789849404 AND 1354857790702 AND p2.gender = 'female'; -- || 71707

-- q11_p016
-- predicates: fcp.creationdate BETWEEN 1308052370341 AND 1352177414840 AND p1.language = 'de;en'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fcp.creationdate BETWEEN 1308052370341 AND 1352177414840 AND p1.language = 'de;en'; -- || 8713

-- q11_p017
-- predicates: plp2.explicitlydeleted = false AND f.title = 'Wall of Rahul Khan' AND fcp.explicitlydeleted = false
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE plp2.explicitlydeleted = false AND f.title = 'Wall of Rahul Khan' AND fcp.explicitlydeleted = false; -- || 4

-- q11_p018
-- predicates: fcp.creationdate BETWEEN 1308052370341 AND 1352177414840 AND p2.gender = 'male'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fcp.creationdate BETWEEN 1308052370341 AND 1352177414840 AND p2.gender = 'male'; -- || 63531

-- q11_p019
-- predicates: plp2.creationdate BETWEEN 1321581133136 AND 1354960011366 AND po.explicitlydeleted = false AND p2.language = 'es;en'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE plp2.creationdate BETWEEN 1321581133136 AND 1354960011366 AND po.explicitlydeleted = false AND p2.language = 'es;en'; -- || 7318

-- q11_p020
-- predicates: plp2.explicitlydeleted = false AND p2.gender = 'male' AND fcp.explicitlydeleted = false
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE plp2.explicitlydeleted = false AND p2.gender = 'male' AND fcp.explicitlydeleted = false; -- || 80223

-- q11_p021
-- predicates: plp1.creationdate BETWEEN 1321581133136 AND 1354960011366 AND po.language = 'en'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE plp1.creationdate BETWEEN 1321581133136 AND 1354960011366 AND po.language = 'en'; -- || 32674

-- q11_p022
-- predicates: pkp.src >= 253343 AND p1.language = 'es;en' AND fhm2.explicitlydeleted = false
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE pkp.src >= 253343 AND p1.language = 'es;en' AND fhm2.explicitlydeleted = false; -- || 3670

-- q11_p023
-- predicates: fcp.explicitlydeleted = false AND p2.language = 'zh;en'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fcp.explicitlydeleted = false AND p2.language = 'zh;en'; -- || 21801

-- q11_p024
-- predicates: pkp.dst <= 253343 AND p1.gender = 'male' AND fhm1.creationdate BETWEEN 1321789849404 AND 1354857790702
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE pkp.dst <= 253343 AND p1.gender = 'male' AND fhm1.creationdate BETWEEN 1321789849404 AND 1354857790702; -- || 37553

-- q11_p025
-- predicates: pkp.dst <= 253343 AND po.creationdate BETWEEN 1308052370341 AND 1352177414840
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE pkp.dst <= 253343 AND po.creationdate BETWEEN 1308052370341 AND 1352177414840; -- || 86468

-- q11_p026
-- predicates: fcp.creationdate BETWEEN 1308052370341 AND 1352177414840 AND p2.language = 'pt;en'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fcp.creationdate BETWEEN 1308052370341 AND 1352177414840 AND p2.language = 'pt;en'; -- || 6286

-- q11_p027
-- predicates: plp1.creationdate BETWEEN 1321581133136 AND 1354960011366 AND p2.language = 'pt;en' AND fcp.explicitlydeleted = false
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE plp1.creationdate BETWEEN 1321581133136 AND 1354960011366 AND p2.language = 'pt;en' AND fcp.explicitlydeleted = false; -- || 6290

-- q11_p028
-- predicates: fcp.explicitlydeleted = false AND p2.language = 'pt;en' AND pkp.src >= 253343
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fcp.explicitlydeleted = false AND p2.language = 'pt;en' AND pkp.src >= 253343; -- || 3060

-- q11_p029
-- predicates: plp1.creationdate BETWEEN 1321581133136 AND 1354960011366 AND p2.language = 'pt;en' AND po.length BETWEEN 0 AND 0
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE plp1.creationdate BETWEEN 1321581133136 AND 1354960011366 AND p2.language = 'pt;en' AND po.length BETWEEN 0 AND 0; -- || 3838

-- q11_p030
-- predicates: fhm1.explicitlydeleted = false AND f.title = 'Wall of Rahul Khan'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fhm1.explicitlydeleted = false AND f.title = 'Wall of Rahul Khan'; -- || 4

-- q11_p031
-- predicates: plp1.explicitlydeleted = false AND f.title = 'Album 1 of Karan Singh' AND fhm2.explicitlydeleted = false
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE plp1.explicitlydeleted = false AND f.title = 'Album 1 of Karan Singh' AND fhm2.explicitlydeleted = false; -- || 6

-- q11_p032
-- predicates: fhm1.explicitlydeleted = false AND p2.language = 'es;en'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fhm1.explicitlydeleted = false AND p2.language = 'es;en'; -- || 11176

-- q11_p033
-- predicates: fhm2.creationdate BETWEEN 1321789849404 AND 1354857790702 AND p2.language = 'de;en' AND plp2.explicitlydeleted = false
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fhm2.creationdate BETWEEN 1321789849404 AND 1354857790702 AND p2.language = 'de;en' AND plp2.explicitlydeleted = false; -- || 8666

-- q11_p034
-- predicates: fhm1.explicitlydeleted = false AND po.language = 'en'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fhm1.explicitlydeleted = false AND po.language = 'en'; -- || 46459

-- q11_p035
-- predicates: fhm2.explicitlydeleted = false AND p2.language = 'pt;en'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fhm2.explicitlydeleted = false AND p2.language = 'pt;en'; -- || 8864

-- q11_p036
-- predicates: plp1.creationdate BETWEEN 1321581133136 AND 1354960011366 AND po.explicitlydeleted = false
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE plp1.creationdate BETWEEN 1321581133136 AND 1354960011366 AND po.explicitlydeleted = false; -- || 126828

-- q11_p037
-- predicates: pkp.src BETWEEN 252947 AND 253760 AND po.creationdate BETWEEN 1308052370341 AND 1352177414840 AND fhm1.creationdate BETWEEN 1321789849404 AND 1354857790702
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE pkp.src BETWEEN 252947 AND 253760 AND po.creationdate BETWEEN 1308052370341 AND 1352177414840 AND fhm1.creationdate BETWEEN 1321789849404 AND 1354857790702; -- || 71002

-- q11_p038
-- predicates: fcp.creationdate BETWEEN 1308052370341 AND 1352177414840 AND po.length <= 0
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fcp.creationdate BETWEEN 1308052370341 AND 1352177414840 AND po.length <= 0; -- || 79684

-- q11_p039
-- predicates: plp1.explicitlydeleted = false AND p1.language = 'en' AND pkp.src >= 253343
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE plp1.explicitlydeleted = false AND p1.language = 'en' AND pkp.src >= 253343; -- || 4441

-- q11_p040
-- predicates: pkp.dst BETWEEN 252947 AND 253760 AND po.length <= 0
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE pkp.dst BETWEEN 252947 AND 253760 AND po.length <= 0; -- || 61703

-- q11_p041
-- predicates: plp2.creationdate BETWEEN 1321581133136 AND 1354960011366 AND po.creationdate BETWEEN 1308052370341 AND 1352177414840
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE plp2.creationdate BETWEEN 1321581133136 AND 1354960011366 AND po.creationdate BETWEEN 1308052370341 AND 1352177414840; -- || 116214

-- q11_p042
-- predicates: fhm2.creationdate BETWEEN 1321789849404 AND 1354857790702 AND po.explicitlydeleted = false AND fcp.creationdate BETWEEN 1308052370341 AND 1352177414840
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fhm2.creationdate BETWEEN 1321789849404 AND 1354857790702 AND po.explicitlydeleted = false AND fcp.creationdate BETWEEN 1308052370341 AND 1352177414840; -- || 108054

-- q11_p043
-- predicates: plp2.explicitlydeleted = false AND p1.language = 'pt;en'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE plp2.explicitlydeleted = false AND p1.language = 'pt;en'; -- || 8677

-- q11_p044
-- predicates: plp1.creationdate BETWEEN 1321581133136 AND 1354960011366 AND p2.language = 'de;en' AND plp2.explicitlydeleted = false
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE plp1.creationdate BETWEEN 1321581133136 AND 1354960011366 AND p2.language = 'de;en' AND plp2.explicitlydeleted = false; -- || 8297

-- q11_p045
-- predicates: plp1.creationdate BETWEEN 1321581133136 AND 1354960011366 AND f.creationdate BETWEEN 1305589168419 AND 1351610285871 AND p1.language = 'en'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE plp1.creationdate BETWEEN 1321581133136 AND 1354960011366 AND f.creationdate BETWEEN 1305589168419 AND 1351610285871 AND p1.language = 'en'; -- || 7928

-- q11_p046
-- predicates: fcp.creationdate BETWEEN 1308052370341 AND 1352177414840 AND p1.language = 'de;en' AND po.explicitlydeleted = false
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fcp.creationdate BETWEEN 1308052370341 AND 1352177414840 AND p1.language = 'de;en' AND po.explicitlydeleted = false; -- || 8359

-- q11_p047
-- predicates: fhm1.explicitlydeleted = false AND po.length >= 0 AND fcp.creationdate BETWEEN 1308052370341 AND 1352177414840
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fhm1.explicitlydeleted = false AND po.length >= 0 AND fcp.creationdate BETWEEN 1308052370341 AND 1352177414840; -- || 136074

-- q11_p048
-- predicates: pkp.dst BETWEEN 252947 AND 253760 AND p1.language = 'pt;en'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE pkp.dst BETWEEN 252947 AND 253760 AND p1.language = 'pt;en'; -- || 5230

-- q11_p049
-- predicates: plp1.creationdate BETWEEN 1321581133136 AND 1354960011366 AND p1.gender = 'female'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE plp1.creationdate BETWEEN 1321581133136 AND 1354960011366 AND p1.gender = 'female'; -- || 69845

-- q11_p050
-- predicates: plp1.creationdate BETWEEN 1321581133136 AND 1354960011366 AND f.creationdate BETWEEN 1305589168419 AND 1351610285871
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE plp1.creationdate BETWEEN 1321581133136 AND 1354960011366 AND f.creationdate BETWEEN 1305589168419 AND 1351610285871; -- || 95358

-- q11_p051
-- predicates: pkp.dst <= 253343 AND p1.language = 'pt;en' AND fcp.explicitlydeleted = false
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE pkp.dst <= 253343 AND p1.language = 'pt;en' AND fcp.explicitlydeleted = false; -- || 5689

-- q11_p052
-- predicates: fcp.explicitlydeleted = false AND po.length BETWEEN 0 AND 0 AND p1.gender = 'female'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fcp.explicitlydeleted = false AND po.length BETWEEN 0 AND 0 AND p1.gender = 'female'; -- || 56796

-- q11_p053
-- predicates: fhm2.explicitlydeleted = false AND p2.gender = 'female'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fhm2.explicitlydeleted = false AND p2.gender = 'female'; -- || 97909

-- q11_p054
-- predicates: plp2.creationdate BETWEEN 1321581133136 AND 1354960011366 AND p1.language = 'de;en'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE plp2.creationdate BETWEEN 1321581133136 AND 1354960011366 AND p1.language = 'de;en'; -- || 8603

-- q11_p055
-- predicates: plp2.explicitlydeleted = false AND po.language = 'en' AND p2.language = 'en'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE plp2.explicitlydeleted = false AND po.language = 'en' AND p2.language = 'en'; -- || 3442

-- q11_p056
-- predicates: pkp.dst BETWEEN 252947 AND 253760 AND f.creationdate BETWEEN 1305589168419 AND 1351610285871
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE pkp.dst BETWEEN 252947 AND 253760 AND f.creationdate BETWEEN 1305589168419 AND 1351610285871; -- || 70089

-- q11_p057
-- predicates: pkp.src BETWEEN 252947 AND 253760 AND f.title = 'Album 1 of Karan Singh' AND plp2.creationdate BETWEEN 1321581133136 AND 1354960011366
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE pkp.src BETWEEN 252947 AND 253760 AND f.title = 'Album 1 of Karan Singh' AND plp2.creationdate BETWEEN 1321581133136 AND 1354960011366; -- || 1

-- q11_p058
-- predicates: fhm1.creationdate BETWEEN 1321789849404 AND 1354857790702 AND p2.gender = 'female' AND f.creationdate BETWEEN 1305589168419 AND 1351610285871
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fhm1.creationdate BETWEEN 1321789849404 AND 1354857790702 AND p2.gender = 'female' AND f.creationdate BETWEEN 1305589168419 AND 1351610285871; -- || 51426

-- q11_p059
-- predicates: fhm1.explicitlydeleted = false AND p1.gender = 'male'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fhm1.explicitlydeleted = false AND p1.gender = 'male'; -- || 83954

-- q11_p060
-- predicates: pkp.src BETWEEN 252947 AND 253760 AND po.length >= 0
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE pkp.src BETWEEN 252947 AND 253760 AND po.length >= 0; -- || 106978

-- q11_p061
-- predicates: pkp.src BETWEEN 252947 AND 253760 AND f.explicitlydeleted = false AND p1.language = 'en'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE pkp.src BETWEEN 252947 AND 253760 AND f.explicitlydeleted = false AND p1.language = 'en'; -- || 10400

-- q11_p062
-- predicates: fhm2.explicitlydeleted = false AND po.length >= 0
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fhm2.explicitlydeleted = false AND po.length >= 0; -- || 181863

-- q11_p063
-- predicates: pkp.src BETWEEN 252947 AND 253760 AND p2.language = 'en'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE pkp.src BETWEEN 252947 AND 253760 AND p2.language = 'en'; -- || 8485

-- q11_p064
-- predicates: pkp.src >= 253343 AND p1.language = 'zh;en'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE pkp.src >= 253343 AND p1.language = 'zh;en'; -- || 8642

-- q11_p065
-- predicates: pkp.dst BETWEEN 252947 AND 253760 AND f.explicitlydeleted = false
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE pkp.dst BETWEEN 252947 AND 253760 AND f.explicitlydeleted = false; -- || 106466

-- q11_p066
-- predicates: pkp.src >= 253343 AND p2.language = 'pt;en'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE pkp.src >= 253343 AND p2.language = 'pt;en'; -- || 3119

-- q11_p067
-- predicates: fhm1.explicitlydeleted = false AND p2.language = 'en' AND fhm2.explicitlydeleted = false
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fhm1.explicitlydeleted = false AND p2.language = 'en' AND fhm2.explicitlydeleted = false; -- || 14527

-- q11_p068
-- predicates: pkp.dst <= 253343 AND po.length <= 0 AND fcp.explicitlydeleted = false
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE pkp.dst <= 253343 AND po.length <= 0 AND fcp.explicitlydeleted = false; -- || 66843

-- q11_p069
-- predicates: fhm2.creationdate BETWEEN 1321789849404 AND 1354857790702 AND po.creationdate BETWEEN 1308052370341 AND 1352177414840 AND p2.language = 'de;en'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fhm2.creationdate BETWEEN 1321789849404 AND 1354857790702 AND po.creationdate BETWEEN 1308052370341 AND 1352177414840 AND p2.language = 'de;en'; -- || 7609

-- q11_p070
-- predicates: plp2.creationdate BETWEEN 1321581133136 AND 1354960011366 AND p2.language = 'en'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE plp2.creationdate BETWEEN 1321581133136 AND 1354960011366 AND p2.language = 'en'; -- || 10661

-- q11_p071
-- predicates: fcp.creationdate BETWEEN 1308052370341 AND 1352177414840 AND p2.language = 'es;en'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fcp.creationdate BETWEEN 1308052370341 AND 1352177414840 AND p2.language = 'es;en'; -- || 8296

-- q11_p072
-- predicates: fhm1.explicitlydeleted = false AND p1.gender = 'female' AND plp1.explicitlydeleted = false
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fhm1.explicitlydeleted = false AND p1.gender = 'female' AND plp1.explicitlydeleted = false; -- || 95985

-- q11_p073
-- predicates: fhm2.creationdate BETWEEN 1321789849404 AND 1354857790702 AND po.creationdate BETWEEN 1308052370341 AND 1352177414840
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fhm2.creationdate BETWEEN 1321789849404 AND 1354857790702 AND po.creationdate BETWEEN 1308052370341 AND 1352177414840; -- || 111944

-- q11_p074
-- predicates: plp2.explicitlydeleted = false AND p2.language = 'de;en' AND fhm2.explicitlydeleted = false
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE plp2.explicitlydeleted = false AND p2.language = 'de;en' AND fhm2.explicitlydeleted = false; -- || 11298

-- q11_p075
-- predicates: fcp.explicitlydeleted = false AND p2.language = 'pt;en'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fcp.explicitlydeleted = false AND p2.language = 'pt;en'; -- || 8747

-- q11_p076
-- predicates: plp1.explicitlydeleted = false AND p1.language = 'de;en' AND p2.language = 'zh;en'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE plp1.explicitlydeleted = false AND p1.language = 'de;en' AND p2.language = 'zh;en'; -- || 629

-- q11_p077
-- predicates: plp1.creationdate BETWEEN 1321581133136 AND 1354960011366 AND po.length BETWEEN 0 AND 0
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE plp1.creationdate BETWEEN 1321581133136 AND 1354960011366 AND po.length BETWEEN 0 AND 0; -- || 77133

-- q11_p078
-- predicates: pkp.src BETWEEN 252947 AND 253760 AND p2.language = 'en' AND fhm1.explicitlydeleted = false
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE pkp.src BETWEEN 252947 AND 253760 AND p2.language = 'en' AND fhm1.explicitlydeleted = false; -- || 8434

-- q11_p079
-- predicates: pkp.src BETWEEN 252947 AND 253760 AND p2.language = 'zh;en'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE pkp.src BETWEEN 252947 AND 253760 AND p2.language = 'zh;en'; -- || 14554

-- q11_p080
-- predicates: pkp.dst <= 253343 AND f.title = 'Album 1 of Karan Singh'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE pkp.dst <= 253343 AND f.title = 'Album 1 of Karan Singh'; -- || 3

-- q11_p081
-- predicates: plp2.creationdate BETWEEN 1321581133136 AND 1354960011366 AND p1.language = 'pt;en'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE plp2.creationdate BETWEEN 1321581133136 AND 1354960011366 AND p1.language = 'pt;en'; -- || 6433

-- q11_p082
-- predicates: fhm2.creationdate BETWEEN 1321789849404 AND 1354857790702 AND p2.gender = 'male' AND po.creationdate BETWEEN 1308052370341 AND 1352177414840
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fhm2.creationdate BETWEEN 1321789849404 AND 1354857790702 AND p2.gender = 'male' AND po.creationdate BETWEEN 1308052370341 AND 1352177414840; -- || 52295

-- q11_p083
-- predicates: plp2.explicitlydeleted = false AND p2.language = 'en' AND p1.language = 'es;en'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE plp2.explicitlydeleted = false AND p2.language = 'en' AND p1.language = 'es;en'; -- || 898

-- q11_p084
-- predicates: fcp.explicitlydeleted = false AND po.length <= 0
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fcp.explicitlydeleted = false AND po.length <= 0; -- || 103646

-- q11_p085
-- predicates: fhm2.creationdate BETWEEN 1321789849404 AND 1354857790702 AND po.creationdate >= 1328021100728
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fhm2.creationdate BETWEEN 1321789849404 AND 1354857790702 AND po.creationdate >= 1328021100728; -- || 112333

-- q11_p086
-- predicates: pkp.src BETWEEN 252947 AND 253760 AND f.title = 'Wall of Rahul Khan'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE pkp.src BETWEEN 252947 AND 253760 AND f.title = 'Wall of Rahul Khan'; -- || 1

-- q11_p087
-- predicates: fhm1.creationdate BETWEEN 1321789849404 AND 1354857790702 AND p2.language = 'pt;en'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fhm1.creationdate BETWEEN 1321789849404 AND 1354857790702 AND p2.language = 'pt;en'; -- || 6356

-- q11_p088
-- predicates: pkp.dst <= 253343 AND po.language = 'en'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE pkp.dst <= 253343 AND po.language = 'en'; -- || 29152

-- q11_p089
-- predicates: fhm1.creationdate BETWEEN 1321789849404 AND 1354857790702 AND f.title = 'Wall of Rahul Khan'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fhm1.creationdate BETWEEN 1321789849404 AND 1354857790702 AND f.title = 'Wall of Rahul Khan'; -- || 3

-- q11_p090
-- predicates: fhm2.creationdate BETWEEN 1321789849404 AND 1354857790702 AND p2.gender = 'male'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fhm2.creationdate BETWEEN 1321789849404 AND 1354857790702 AND p2.gender = 'male'; -- || 62349

-- q11_p091
-- predicates: fcp.explicitlydeleted = false AND p1.gender = 'female' AND fhm1.explicitlydeleted = false
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fcp.explicitlydeleted = false AND p1.gender = 'female' AND fhm1.explicitlydeleted = false; -- || 95796

-- q11_p092
-- predicates: fcp.explicitlydeleted = false AND f.creationdate BETWEEN 1305589168419 AND 1351610285871
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fcp.explicitlydeleted = false AND f.creationdate BETWEEN 1305589168419 AND 1351610285871; -- || 110280

-- q11_p093
-- predicates: fhm2.explicitlydeleted = false AND p2.gender = 'male'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fhm2.explicitlydeleted = false AND p2.gender = 'male'; -- || 83954

-- q11_p094
-- predicates: pkp.src BETWEEN 252947 AND 253760 AND f.title = 'Album 1 of Karan Singh' AND fhm1.creationdate BETWEEN 1321789849404 AND 1354857790702
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE pkp.src BETWEEN 252947 AND 253760 AND f.title = 'Album 1 of Karan Singh' AND fhm1.creationdate BETWEEN 1321789849404 AND 1354857790702; -- || 1

-- q11_p095
-- predicates: plp1.explicitlydeleted = false AND p1.language = 'es;en'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE plp1.explicitlydeleted = false AND p1.language = 'es;en'; -- || 10904

-- q11_p096
-- predicates: fhm1.explicitlydeleted = false AND p1.language = 'es;en'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fhm1.explicitlydeleted = false AND p1.language = 'es;en'; -- || 11178

-- q11_p097
-- predicates: plp1.creationdate BETWEEN 1321581133136 AND 1354960011366 AND f.title = 'Album 1 of Karan Singh' AND po.creationdate >= 1328021100728
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE plp1.creationdate BETWEEN 1321581133136 AND 1354960011366 AND f.title = 'Album 1 of Karan Singh' AND po.creationdate >= 1328021100728; -- || 6

-- q11_p098
-- predicates: pkp.src >= 253343 AND p1.language = 'pt;en'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE pkp.src >= 253343 AND p1.language = 'pt;en'; -- || 2455

-- q11_p099
-- predicates: pkp.dst BETWEEN 252947 AND 253760 AND p1.language = 'es;en'
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE pkp.dst BETWEEN 252947 AND 253760 AND p1.language = 'es;en'; -- || 5985

-- q11_p100
-- predicates: fhm2.explicitlydeleted = false AND p1.language = 'zh;en' AND f.explicitlydeleted = false
SELECT
  -- 点表
  f.*,
  po.*,
  p1.*,
  p2.*,

  -- 边表
  fcp.*,
  fhm1.*,
  fhm2.*,
  pkp.*,
  plp1.*,
  plp2.*
FROM forum AS f
JOIN forum_containerof_post AS fcp
  ON fcp.forumid = f.id
JOIN post AS po
  ON po.id = fcp.postid
JOIN forum_hasmember_person AS fhm1
  ON fhm1.forumid = f.id
JOIN person AS p1
  ON p1.id = fhm1.personid
JOIN forum_hasmember_person AS fhm2
  ON fhm2.forumid = f.id
JOIN person AS p2
  ON p2.id = fhm2.personid
JOIN person_knows_person AS pkp
  ON pkp.src = p1.id
 AND pkp.dst = p2.id
JOIN person_likes_post AS plp1
  ON plp1.personid = p1.id
 AND plp1.postid = po.id
JOIN person_likes_post AS plp2
  ON plp2.personid = p2.id
 AND plp2.postid = po.id
WHERE fhm2.explicitlydeleted = false AND p1.language = 'zh;en' AND f.explicitlydeleted = false; -- || 22062

