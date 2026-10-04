# Restore the Atlas free cluster from a nightly backup

The free cluster has no backups; `.github/workflows/backup.yml` dumps the `docqa`
database every night and keeps the archive for 14 days.

1. GitHub → Actions → **backup** → the latest successful run → download `docqa-backup`.
2. Unzip it: you get `docqa-YYYY-MM-DD.archive.gz`.
3. Restore into the cluster (drops and replaces each collection in the dump):

   ```bash
   mongorestore --uri "$MONGODB_URI_ATLAS" --gzip --archive=docqa-YYYY-MM-DD.archive.gz --drop
   ```

4. Rebuild the search indexes (they are not part of a dump) and wait for READY:

   ```bash
   MONGODB_URI=$MONGODB_URI_ATLAS make initdb
   ```

5. Check: `GET /health` returns 200 and both indexes are READY; ask one known question.

Restore drill: do steps 1–5 against atlas-local once before the demo and note the time taken.
