"""
Database utilities for Hexa Paint Animation backend
"""

import sqlite3
import os
from datetime import datetime
from typing import Tuple, Dict, Any

class DatabaseManager:
    """Database manager for usage statistics"""
    
    def __init__(self, db_path: str = 'usage_stats.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create usage_stats table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS usage_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_ip TEXT,
                filename TEXT,
                processing_time REAL,
                success BOOLEAN
            )
        ''')
        
        # Create index for better performance
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON usage_stats(timestamp)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_success 
            ON usage_stats(success)
        ''')
        
        conn.commit()
        conn.close()
    
    def record_usage(self, user_ip: str, filename: str, processing_time: float, success: bool) -> bool:
        """
        Record usage statistics
        
        Args:
            user_ip: User's IP address
            filename: Original filename
            processing_time: Time taken to process
            success: Whether processing was successful
            
        Returns:
            bool: True if successful
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO usage_stats (user_ip, filename, processing_time, success)
                VALUES (?, ?, ?, ?)
            ''', (user_ip, filename, processing_time, success))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Database error: {e}")
            return False
    
    def get_usage_stats(self) -> Tuple[int, int]:
        """
        Get usage statistics
        
        Returns:
            Tuple of (total_uses, successful_uses)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM usage_stats')
            total_uses = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM usage_stats WHERE success = 1')
            successful_uses = cursor.fetchone()[0]
            
            conn.close()
            return total_uses, successful_uses
        except Exception as e:
            print(f"Database error: {e}")
            return 0, 0
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """
        Get detailed usage statistics
        
        Returns:
            Dictionary with detailed statistics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total uses
            cursor.execute('SELECT COUNT(*) FROM usage_stats')
            total_uses = cursor.fetchone()[0]
            
            # Successful uses
            cursor.execute('SELECT COUNT(*) FROM usage_stats WHERE success = 1')
            successful_uses = cursor.fetchone()[0]
            
            # Average processing time
            cursor.execute('SELECT AVG(processing_time) FROM usage_stats WHERE success = 1')
            avg_processing_time = cursor.fetchone()[0] or 0
            
            # Recent activity (last 24 hours)
            cursor.execute('''
                SELECT COUNT(*) FROM usage_stats 
                WHERE timestamp > datetime('now', '-1 day')
            ''')
            recent_uses = cursor.fetchone()[0]
            
            # Most common file types
            cursor.execute('''
                SELECT 
                    CASE 
                        WHEN filename LIKE '%.png' THEN 'PNG'
                        WHEN filename LIKE '%.jpg' OR filename LIKE '%.jpeg' THEN 'JPEG'
                        WHEN filename LIKE '%.gif' THEN 'GIF'
                        WHEN filename LIKE '%.bmp' THEN 'BMP'
                        WHEN filename LIKE '%.tiff' THEN 'TIFF'
                        ELSE 'Other'
                    END as file_type,
                    COUNT(*) as count
                FROM usage_stats 
                WHERE success = 1
                GROUP BY file_type
                ORDER BY count DESC
            ''')
            file_types = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                'total_uses': total_uses,
                'successful_uses': successful_uses,
                'failed_uses': total_uses - successful_uses,
                'success_rate': round((successful_uses / total_uses * 100) if total_uses > 0 else 0, 2),
                'avg_processing_time': round(avg_processing_time, 2),
                'recent_uses_24h': recent_uses,
                'file_types': file_types
            }
        except Exception as e:
            print(f"Database error: {e}")
            return {
                'total_uses': 0,
                'successful_uses': 0,
                'failed_uses': 0,
                'success_rate': 0,
                'avg_processing_time': 0,
                'recent_uses_24h': 0,
                'file_types': {}
            }
    
    def cleanup_old_records(self, days: int = 30) -> int:
        """
        Clean up old records to keep database size manageable
        
        Args:
            days: Number of days to keep records
            
        Returns:
            Number of records deleted
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                DELETE FROM usage_stats 
                WHERE timestamp < datetime('now', '-{} days')
            '''.format(days))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            return deleted_count
        except Exception as e:
            print(f"Database cleanup error: {e}")
            return 0
    
    def get_database_size(self) -> int:
        """
        Get database file size in bytes
        
        Returns:
            Database file size in bytes
        """
        try:
            if os.path.exists(self.db_path):
                return os.path.getsize(self.db_path)
            return 0
        except Exception as e:
            print(f"Error getting database size: {e}")
            return 0
