#!/bin/bash

# PostgreSQL Vector Database Management Script

case "$1" in
    start)
        echo "Starting PostgreSQL Vector Database..."
        docker-compose up -d postgresql-vector
        echo "Database started. Waiting for it to become ready..."
        docker-compose exec postgresql-vector bash -c 'until pg_isready -U postgres -d vectordb; do sleep 1; done'
        echo "PostgreSQL Vector Database is ready!"
        ;;
    stop)
        echo "Stopping PostgreSQL Vector Database..."
        docker-compose stop postgresql-vector
        echo "Database stopped."
        ;;
    restart)
        echo "Restarting PostgreSQL Vector Database..."
        docker-compose restart postgresql-vector
        echo "Database restarted."
        ;;
    status)
        echo "PostgreSQL Vector Database Status:"
        docker-compose ps postgresql-vector
        ;;
    logs)
        echo "PostgreSQL Vector Database Logs:"
        docker-compose logs postgresql-vector
        ;;
    shell)
        echo "Connecting to PostgreSQL shell..."
        docker-compose exec postgresql-vector psql -U postgres -d vectordb
        ;;
    build)
        echo "Building PostgreSQL Vector Database image..."
        docker-compose build postgresql-vector
        ;;
    clean)
        echo "Stopping and removing PostgreSQL Vector Database..."
        docker-compose down postgresql-vector
        echo "Removing database volume (WARNING: This will delete all data!)..."
        read -p "Are you sure you want to delete all database data? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker volume rm vectordb-plugin_postgres_data
            echo "Database volume removed."
        else
            echo "Volume removal cancelled."
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|shell|build|clean}"
        echo ""
        echo "Commands:"
        echo "  start   - Start the PostgreSQL vector database"
        echo "  stop    - Stop the PostgreSQL vector database"
        echo "  restart - Restart the PostgreSQL vector database"
        echo "  status  - Show database container status"
        echo "  logs    - Show database container logs"
        echo "  shell   - Connect to PostgreSQL shell"
        echo "  build   - Build the database image"
        echo "  clean   - Stop and remove database (WARNING: Deletes all data)"
        echo ""
        echo "Configuration from config.yaml:"
        echo "  Database: vectordb"
        echo "  Host: localhost"
        echo "  Port: 5433"
        echo "  User: postgres"
        exit 1
        ;;
esac 