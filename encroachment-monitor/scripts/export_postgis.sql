-- ============================================================================
-- PostGIS storage layer (paper Table I / Table IV)
--
-- Optional production schema for the framework's data-storage layer: raster
-- tiles in cloud object storage, parcel geometry and detections in PostGIS.
-- The repository's demo runs on the file-based AlertStore (dashboard/);
-- load the cadastre and detections produced by run_all.py with these tables:
--
--   psql -d landdb -f scripts/export_postgis.sql
--   psql -d landdb -c "\copy parcels FROM 'results/data/parcels.csv' CSV HEADER"
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS postgis;

-- Cadastral layer (WGS-84 / UTM per Table IV) --------------------------------
CREATE TABLE IF NOT EXISTS parcels (
    id        TEXT PRIMARY KEY,
    name      TEXT NOT NULL,
    protected BOOLEAN NOT NULL,
    area_m2   DOUBLE PRECISION,
    geom      geometry(Polygon, 32643) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS parcels_geom_gix ON parcels USING GIST (geom);
CREATE INDEX IF NOT EXISTS parcels_protected_ix ON parcels (protected) WHERE protected;

-- Raster tile archive (multi-temporal record; Sec. Alerts/Deployment) --------
CREATE TABLE IF NOT EXISTS raster_tiles (
    tile_id    TEXT PRIMARY KEY,          -- e.g. T43PHR_20260103T053651
    scene_crs  TEXT NOT NULL DEFAULT 'EPSG:32643',
    acquired_at TIMESTAMPTZ NOT NULL,
    cloud_pct  REAL,
    object_uri TEXT NOT NULL,             -- s3://tiles/... (object store)
    geom       geometry(Polygon, 32643) NOT NULL
);
CREATE INDEX IF NOT EXISTS tiles_geom_gix ON raster_tiles USING GIST (geom);
CREATE INDEX IF NOT EXISTS tiles_time_ix ON raster_tiles (acquired_at DESC);

-- Detections and the audit trail ---------------------------------------------
CREATE TABLE IF NOT EXISTS detections (
    id           BIGSERIAL PRIMARY KEY,
    site_id      TEXT UNIQUE NOT NULL,
    parcel_id    TEXT REFERENCES parcels(id),
    ndbi_t1      REAL NOT NULL,
    ndbi_t2      REAL NOT NULL,
    d_ndbi       REAL NOT NULL,
    area_m2      REAL,
    severity     TEXT CHECK (severity IN ('low','medium','high')),
    is_true_positive BOOLEAN,
    detected_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    geom         geometry(Point, 32643) NOT NULL
);
CREATE INDEX IF NOT EXISTS det_geom_gix ON detections USING GIST (geom);

-- The GIS overlay as a spatial query: new construction ∩ protected parcels ---
CREATE OR REPLACE VIEW v_encroachments AS
SELECT d.*, p.name AS parcel_name
FROM detections d
JOIN parcels p ON p.id = d.parcel_id AND p.protected;

-- Audit log (Sec. Alerts, Security and Deployment) ---------------------------
CREATE TABLE IF NOT EXISTS audit_log (
    id     BIGSERIAL PRIMARY KEY,
    ts     TIMESTAMPTZ NOT NULL DEFAULT now(),
    user_name TEXT NOT NULL,
    role   TEXT NOT NULL,
    action TEXT NOT NULL,
    detail TEXT
);

-- Escalation sweep: alerts unacknowledged past the SLA (72 h default) --------
CREATE OR REPLACE FUNCTION escalate_stale_alerts(sla_hours DOUBLE PRECISION DEFAULT 72)
RETURNS INTEGER AS $$
DECLARE
    n INTEGER;
BEGIN
    UPDATE alerts
       SET status = 'ESCALATED'
     WHERE status IN ('NEW', 'ACKNOWLEDGED')
       AND created_at < now() - make_interval(hours => sla_hours);
    GET DIAGNOSTICS n = ROW_COUNT;
    INSERT INTO audit_log(user_name, role, action, detail)
    VALUES ('system', 'admin', 'escalate', format('SLA sweep escalated %s alerts', n));
    RETURN n;
END;
$$ LANGUAGE plpgsql;

-- Announce success
DO $$ BEGIN RAISE NOTICE 'PostGIS schema ready: parcels, raster_tiles, detections, audit_log, escalate_stale_alerts()'; END $$;
