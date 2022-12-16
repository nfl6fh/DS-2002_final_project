import pandas as pd
import pymongo

def main():
    print('Ingesting data...')
    # Connect to MongoDB
    client = pymongo.MongoClient('mongodb://localhost:27017/')
    db = client['netflix']
    # Ingest data
    shows = pd.read_csv('data/Best Shows Netflix.csv', index_col=0)
    show_yr = pd.read_csv('data/Best Show by Year Netflix.csv', index_col=0)
    movies = pd.read_csv('data/Best Movies Netflix.csv', index_col=0)
    movie_yr = pd.read_csv('data/Best Movie by Year Netflix.csv', index_col=0)

    # remove irrelevant data (not from 2017-2022)
    shows = shows.where(shows.RELEASE_YEAR >= 2017).dropna()[['TITLE', 'SCORE', 'RELEASE_YEAR']].reset_index(drop=True)
    movies = movies.where(movies.RELEASE_YEAR >= 2017).dropna()[['TITLE', 'SCORE', 'RELEASE_YEAR']].reset_index(drop=True)
    show_yr = show_yr.where(show_yr.RELEASE_YEAR >= 2017).dropna()[['TITLE', 'SCORE', 'RELEASE_YEAR']].reset_index(drop=True)
    movie_yr = movie_yr.where(movie_yr.RELEASE_YEAR >= 2017).dropna()[['TITLE', 'SCORE', 'RELEASE_YEAR']].reset_index(drop=True)

    # clear collections
    db.shows.drop()
    db.movies.drop()
    db.show_yr.drop()
    db.movie_yr.drop()

    # add data to MongoDB
    db.shows.insert_many(shows.to_dict('records'))
    db.movies.insert_many(movies.to_dict('records'))
    db.show_yr.insert_many(show_yr.to_dict('records'))
    db.movie_yr.insert_many(movie_yr.to_dict('records'))
    return

if __name__ == '__main__':
    main()