#!/bin/bash
set -eu
set -o pipefail

# Build duckdb database from LDBC SNB dataset with explicit column definitions
# Argument: sf

sf=$1

workspace=$(realpath $(dirname $0)/../../)
duckdb=$workspace/tools/duckdb

mkdir -p $workspace/graphs/ldbc/duckdb/
cd $workspace/datasets/ldbc/sf$sf
rm -f "$workspace/graphs/ldbc/duckdb/ldbc_with_attrs_sf$sf.duckdb"

# Use Python to generate SQL with column definitions
python3 <<PYTHON | $duckdb "$workspace/graphs/ldbc/duckdb/ldbc_with_attrs_sf$sf.duckdb"
import csv
import os
import sys

def get_column_defs(csv_file):
    """Read CSV header and return column definitions"""
    if not os.path.exists(csv_file):
        print(f"Error: CSV file {csv_file} not found", file=sys.stderr)
        return None
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
    
    # List column names with VARCHAR type
    # DuckDB will handle type conversion during read_csv with auto_detect
    # Using VARCHAR as safe default - DuckDB can optimize later
    col_defs = []
    for col in header:
        col = col.strip()
        if col:
            col_defs.append(f'"{col}" VARCHAR')
    
    return ', '.join(col_defs)

def create_table_sql(table_name, csv_file):
    """Generate CREATE TABLE, INSERT, and INDEX SQL statements"""
    col_defs = get_column_defs(csv_file)
    if col_defs is None:
        return None
    
    # Get column names to determine if it's an edge or vertex table
    if not os.path.exists(csv_file):
        return None
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = [col.strip() for col in next(reader)]
    
    # DuckDB doesn't support CREATE TABLE ... AS SELECT with explicit column definitions
    # Use two-step approach: CREATE TABLE then INSERT
    create_sql = f'create table {table_name} ({col_defs});'
    insert_sql = f'insert into {table_name} select * from read_csv(\'{csv_file}\', header=true, auto_detect=true);'
    
    # Create indexes
    index_sqls = []
    if 'src' in header and 'dst' in header:
        # Edge table: create indexes on src and dst
        index_sqls.append(f'create index idx_{table_name}_src on {table_name}(src);')
        index_sqls.append(f'create index idx_{table_name}_dst on {table_name}(dst);')
    elif 'id' in header:
        # Vertex table: create index on id
        index_sqls.append(f'create index idx_{table_name}_id on {table_name}(id);')
    
    all_sqls = [create_sql, insert_sql] + index_sqls
    return '\n'.join(all_sqls)

# Generate SQL
tables = [
    # Vertex tables
    ("City", "City.csv"),
    ("Comment", "Comment.csv"),
    ("Company", "Company.csv"),
    ("Continent", "Continent.csv"),
    ("Country", "Country.csv"),
    ("Forum", "Forum.csv"),
    ("Person", "Person.csv"),
    ("Post", "Post.csv"),
    ("Tag", "Tag.csv"),
    ("TagClass", "TagClass.csv"),
    ("University", "University.csv"),
    # Edge tables
    ("City_isPartOf_Country", "City_isPartOf_Country.csv"),
    ("Comment_hasCreator_Person", "Comment_hasCreator_Person.csv"),
    ("Comment_hasTag_Tag", "Comment_hasTag_Tag.csv"),
    ("Comment_isLocatedIn_Country", "Comment_isLocatedIn_Country.csv"),
    ("Comment_replyOf_Comment", "Comment_replyOf_Comment.csv"),
    ("Comment_replyOf_Post", "Comment_replyOf_Post.csv"),
    ("Company_isLocatedIn_Country", "Company_isLocatedIn_Country.csv"),
    ("Country_isPartOf_Continent", "Country_isPartOf_Continent.csv"),
    ("Forum_containerOf_Post", "Forum_containerOf_Post.csv"),
    ("Forum_hasMember_Person", "Forum_hasMember_Person.csv"),
    ("Forum_hasModerator_Person", "Forum_hasModerator_Person.csv"),
    ("Forum_hasTag_Tag", "Forum_hasTag_Tag.csv"),
    ("Person_hasInterest_Tag", "Person_hasInterest_Tag.csv"),
    ("Person_isLocatedIn_City", "Person_isLocatedIn_City.csv"),
    ("Person_knows_Person", "Person_knows_Person.csv"),
    ("Person_likes_Comment", "Person_likes_Comment.csv"),
    ("Person_likes_Post", "Person_likes_Post.csv"),
    ("Person_studyAt_University", "Person_studyAt_University.csv"),
    ("Person_workAt_Company", "Person_workAt_Company.csv"),
    ("Post_hasCreator_Person", "Post_hasCreator_Person.csv"),
    ("Post_hasTag_Tag", "Post_hasTag_Tag.csv"),
    ("Post_isLocatedIn_Country", "Post_isLocatedIn_Country.csv"),
    ("TagClass_isSubclassOf_TagClass", "TagClass_isSubclassOf_TagClass.csv"),
    ("Tag_hasType_TagClass", "Tag_hasType_TagClass.csv"),
    ("University_isLocatedIn_City", "University_isLocatedIn_City.csv"),
]

print("begin;")
for table_name, csv_file in tables:
    sql = create_table_sql(table_name, csv_file)
    if sql:
        print(sql)
print("commit;")
PYTHON
